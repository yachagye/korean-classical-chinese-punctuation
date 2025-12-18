"""
Lightning.ai L40S 48GB용 한국 고전한문 구두점 예측 모델 학습
7개 구두점 버전 - 메모리 최적화 버전
"""

import json
import torch
import torch.nn as nn

torch.set_float32_matmul_precision('high')
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class CompressedMultiLabelDataset(Dataset):
    """압축된 다중 라벨 구두점 예측 데이터셋 - 메모리 효율적 버전"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = Path(data_path)

        # 7개 구두점 (학습데이터 생성 코드와 일치)
        self.punctuations = [
            ',', '。', '·', '?', '!', '《', '》'
        ]
        self.num_labels = 7

        print(f"데이터 인덱싱: {data_path}")

        # 파일의 각 라인 위치만 저장 (메모리 효율적)
        self.line_offsets = []

        with open(data_path, 'rb') as f:
            offset = 0
            for line in tqdm(f, desc="Building index"):
                self.line_offsets.append(offset)
                offset = f.tell()

        self.num_samples = len(self.line_offsets)
        print(f"인덱싱 완료: {self.num_samples}개 샘플")

        # 첫 번째 샘플로 데이터 형식 검증
        if self.num_samples > 0:
            sample = self._load_sample(0)
            required_keys = ['c', 'l', 'n']
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"데이터 형식 오류: '{key}' 키가 없습니다. "
                                     f"사용 가능한 키: {list(sample.keys())}")

    def _load_sample(self, idx: int) -> dict:
        """특정 인덱스의 샘플을 파일에서 로드"""
        with open(self.data_path, 'rb') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            return json.loads(line.decode('utf-8').strip())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 필요할 때만 파일에서 데이터 로드
        sample = self._load_sample(idx)

        text = sample['c']  # 2번 파일 키 사용
        length = sample['n']  # 2번 파일 키 사용

        # 압축 라벨 복원 (항상 압축된 형태)
        labels = []
        for indices in sample['l']:  # 2번 파일 키 사용
            label_vec = [0] * self.num_labels  # 7
            for idx_val in indices:
                if 0 <= idx_val < self.num_labels:
                    label_vec[idx_val] = 1
            labels.append(label_vec)

        # 패딩
        while len(labels) < self.max_length:
            labels.append([0] * self.num_labels)
        labels = labels[:self.max_length]

        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 라벨 정렬
        aligned_labels = self._align_labels(text, labels, length, encoding)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.float),
            'length': length
        }

    def _align_labels(self, text: str, labels: List[List[int]], length: int,
                      encoding) -> List[List[float]]:
        """토큰과 라벨 정렬"""
        aligned_labels = [[0.0] * self.num_labels for _ in range(self.max_length)]
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        char_idx = 0
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token == '[UNK]' or not token.startswith('##'):
                if char_idx < length and char_idx < len(labels):
                    # 라벨 값 검증 후 변환
                    aligned_labels[i] = [float(x) for x in labels[char_idx]]
                    char_idx += 1
                elif char_idx >= length:
                    # 실제 길이를 초과하면 중단
                    break

        return aligned_labels


class PatternAwareLoss(nn.Module):
    """패턴을 고려한 손실 함수"""

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # 빈번한 패턴에 가중치
        self.pattern_boost = 1.2

    def forward(self, logits, labels, attention_mask):
        # 기본 BCE 손실
        loss = self.bce_loss(logits, labels)

        # 연속 구두점 위치에 가중치
        batch_size, seq_len, num_labels = labels.shape
        for i in range(seq_len - 1):
            # 연속된 구두점이 있는 위치
            curr_has_punct = labels[:, i, :].sum(dim=1) > 0
            next_has_punct = labels[:, i + 1, :].sum(dim=1) > 0

            pattern_mask = (curr_has_punct & next_has_punct).float()
            if pattern_mask.sum() > 0:
                loss[:, i, :] = loss[:, i, :] * (1 + pattern_mask.unsqueeze(1) * (self.pattern_boost - 1))

        # 마스킹 및 평균
        mask = attention_mask.unsqueeze(-1).float()
        loss = (loss * mask).sum() / mask.sum()

        return loss


class MultiLabelPunctuationModel(pl.LightningModule):
    """PyTorch Lightning 다중 라벨 구두점 예측 모델"""

    def __init__(
            self,
            model_name: str = 'hfl/chinese-roberta-wwm-ext',
            num_labels: int = 7,  # 수정: 7개 구두점
            learning_rate: float = 2e-5,
            warmup_ratio: float = 0.1,
            dropout_rate: float = 0.1,
            threshold: float = 0.5,
            total_steps: Optional[int] = None,
            use_pattern_loss: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        # BERT 모델
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # 손실 함수
        if use_pattern_loss:
            self.loss_fn = PatternAwareLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # 평가용
        self.threshold = threshold
        self.punctuations = [
            ',', '。', '·', '?', '!', '《', '》'
        ]

        # 메트릭 저장 (메모리 효율적 버전)
        self.val_metrics = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return logits

    def compute_loss(self, logits, labels, attention_mask):
        """손실 계산"""
        if hasattr(self.loss_fn, 'forward'):
            # PatternAwareLoss
            return self.loss_fn(logits, labels, attention_mask)
        else:
            # 기본 BCE
            loss = self.loss_fn(
                logits.view(-1, self.hparams.num_labels),
                labels.view(-1, self.hparams.num_labels)
            )
            mask = attention_mask.view(-1, 1).float()
            loss = (loss * mask).sum() / mask.sum()
            return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels, attention_mask)

        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels, attention_mask)

        # 메모리 효율적인 메트릭 계산
        if self.val_metrics is None:
            self.val_metrics = {
                'losses': [],
                'total_samples': 0,
                'label_tp': torch.zeros(self.hparams.num_labels, device=self.device),
                'label_fp': torch.zeros(self.hparams.num_labels, device=self.device),
                'label_fn': torch.zeros(self.hparams.num_labels, device=self.device)
            }

        # 손실만 저장 (텐서가 아닌 스칼라 값으로)
        self.val_metrics['losses'].append(loss.item())

        # 예측값 계산
        preds = torch.sigmoid(logits)
        preds_binary = (preds > self.threshold).float()

        # 마스크 적용
        valid_mask = attention_mask.unsqueeze(-1).bool()
        preds_masked = preds_binary * valid_mask
        labels_masked = labels * valid_mask

        # 라벨별 TP, FP, FN 누적 (GPU에서 계산 후 결과만 저장)
        for i in range(self.hparams.num_labels):
            self.val_metrics['label_tp'][i] += ((preds_masked[:, :, i] == 1) & (labels_masked[:, :, i] == 1)).sum()
            self.val_metrics['label_fp'][i] += ((preds_masked[:, :, i] == 1) & (labels_masked[:, :, i] == 0)).sum()
            self.val_metrics['label_fn'][i] += ((preds_masked[:, :, i] == 0) & (labels_masked[:, :, i] == 1)).sum()

        self.val_metrics['total_samples'] += valid_mask.sum()

        return loss

    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 메트릭 계산"""
        if self.val_metrics is None:
            return

        # 평균 손실
        avg_loss = np.mean(self.val_metrics['losses'])

        # 전체 F1 계산
        tp_sum = self.val_metrics['label_tp'].sum().item()
        fp_sum = self.val_metrics['label_fp'].sum().item()
        fn_sum = self.val_metrics['label_fn'].sum().item()

        overall_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        overall_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                                  overall_precision + overall_recall) > 0 else 0

        # 구두점별 메트릭
        punct_metrics = {}
        for i, punct in enumerate(self.punctuations):
            tp = self.val_metrics['label_tp'][i].item()
            fp = self.val_metrics['label_fp'][i].item()
            fn = self.val_metrics['label_fn'][i].item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            punct_metrics[punct] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # 로깅
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_f1', overall_f1, prog_bar=True)

        # 구두점 로깅
        top_puncts = sorted(
            punct_metrics.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )

        print(f"\n검증 결과 - F1: {overall_f1:.4f}")
        print("7종 구두점:")
        for punct, scores in top_puncts:
            print(f"  {punct}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}")
            self.log(f'val_f1_{punct}', scores['f1'])

        # 메트릭 초기화
        self.val_metrics = None

    def configure_optimizers(self):
        """옵티마이저 설정"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )

        if self.hparams.total_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.total_steps * self.hparams.warmup_ratio),
                num_training_steps=self.hparams.total_steps
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }

        return optimizer


class PunctuationDataModule(pl.LightningDataModule):
    """데이터 모듈"""

    def __init__(
            self,
            data_dir: str,
            tokenizer,
            batch_size: int = 32,
            max_length: int = 512,
            num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self._setup_done = False

    def setup(self, stage: Optional[str] = None):
        if self._setup_done:
            return

        if stage == 'fit' or stage is None:
            # 압축 데이터셋 사용
            self.train_dataset = CompressedMultiLabelDataset(
                self.data_dir / "train.jsonl",
                self.tokenizer,
                self.max_length
            )
            self.val_dataset = CompressedMultiLabelDataset(
                self.data_dir / "val.jsonl",
                self.tokenizer,
                self.max_length
            )
            self._setup_done = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )


def main():
    """메인 함수"""
    # 설정
    config = {
        'model_name': 'hfl/chinese-roberta-wwm-ext',
        'num_labels': 7,  # 수정: 7개 구두점
        'max_length': 512,
        'batch_size': 80,  # L40S 48GB에 최적화
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_ratio': 0.1,
        'dropout_rate': 0.1,
        'threshold': 0.5,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 2,  # 유효 배치 크기 160
        'precision': 'bf16-mixed',  # L40S는 BF16 지원
        'seed': 42,
        'use_pattern_loss': True  # 패턴 손실 함수 사용
    }

    # 시드 설정
    pl.seed_everything(config['seed'])

    # 경로 설정
    data_dir = Path("/teamspace/studios/this_studio/data")
    output_dir = Path("/teamspace/studios/this_studio/model")
    output_dir.mkdir(exist_ok=True)

    # 토크나이저
    print("토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 데이터 모듈
    print("데이터 모듈 준비...")
    data_module = PunctuationDataModule(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_workers=12  # L40S는 더 많은 CPU 코어 활용 가능
    )

    # 총 스텝 계산 (스케줄러용)
    estimated_train_size = 3308365  # '3_학습데이터_검증_구두점7' 결과의 train.jsonl 규모: 학습데이터 총 개수
    steps_per_epoch = estimated_train_size // config['batch_size']
    optimizer_steps_per_epoch = steps_per_epoch // config['accumulate_grad_batches']
    total_steps = optimizer_steps_per_epoch * config['num_epochs']

    # 모델
    print("모델 초기화...")
    model = MultiLabelPunctuationModel(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        dropout_rate=config['dropout_rate'],
        threshold=config['threshold'],
        total_steps=total_steps,
        use_pattern_loss=config['use_pattern_loss']
    )

    # 콜백
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="punct-{epoch:02d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=4,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # 로거
    loggers = [
        TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard_logs"
        ),
        CSVLogger(
            save_dir=output_dir,
            name="csv_logs"
        )
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='gpu',
        devices=1,
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        val_check_interval=0.25,  # 에폭의 25%마다 검증
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        benchmark=True,  # L40S는 안정적이므로 벤치마크 활성화로 속도 향상
        strategy='auto',
        sync_batchnorm=True,
        profiler='simple'
    )

    # 학습
    print("\n" + "=" * 50)
    print("L40S 48GB GPU 학습 시작! (메모리 최적화 버전)")
    print(f"배치 크기: {config['batch_size']}")
    print(f"Gradient Accumulation: {config['accumulate_grad_batches']}")
    print(f"유효 배치 크기: {config['batch_size'] * config['accumulate_grad_batches']}")
    print(f"에폭당 스텝: {steps_per_epoch:,}")
    print(f"에폭당 Optimizer 스텝: {optimizer_steps_per_epoch:,}")
    print(f"총 Optimizer 스텝: {total_steps:,}")
    print(f"Precision: {config['precision']}")
    print(f"패턴 손실 함수: {'사용' if config['use_pattern_loss'] else '미사용'}")
    print(f"구두점 개수: {config['num_labels']}개")
    print("데이터 로딩: 메모리 효율적 인덱싱 방식")
    print("=" * 50 + "\n")

    trainer.fit(model, data_module)

    print("\n학습 완료!")
    print(f"최고 체크포인트: {trainer.checkpoint_callback.best_model_path}")
    print(f"최고 F1 스코어: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()