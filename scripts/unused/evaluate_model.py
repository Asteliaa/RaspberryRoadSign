#!/usr/bin/env python3
"""
Расширенная валидация YOLO модели для детекции дорожных знаков РБ
Включает анализ по классам, confusion matrix, сохранение результатов
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Расширенный оценщик модели YOLO"""

    def __init__(self, model_path, data_yaml, conf_threshold=0.25, output_dir="validation_results"):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.conf_threshold = conf_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if not self.model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not self.data_yaml.exists():
            logger.error(f"data.yaml не найден: {data_yaml}")
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        logger.info(f"Загружаю модель: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # Загрузим имена классов
        self.class_names = self.model.names
        self.num_classes = len(self.class_names)
        logger.info(f"Классов в модели: {self.num_classes}")

    def validate(self, device=0, split="val"):
        """Валидация модели"""
        logger.info(f"Начинаю валидацию на {split}-сете...")

        results = self.model.val(
            data=str(self.data_yaml),
            conf=self.conf_threshold,
            device=device,
            verbose=False
        )

        return results

    def print_global_metrics(self, results):
        """Вывод глобальных метрик"""
        logger.info("\n" + "="*70)
        logger.info("ГЛОБАЛЬНЫЕ МЕТРИКИ ВАЛИДАЦИИ")
        logger.info("="*70)

        if hasattr(results, 'box'):
            logger.info(f"Precision (all): {results.box.p:.4f}")
            logger.info(f"Recall (all):    {results.box.r:.4f}")
            logger.info(f"mAP50:           {results.box.map50:.4f}")
            logger.info(f"mAP50-95:        {results.box.map:.4f}")

        logger.info("="*70)

    def get_per_class_metrics(self, results):
        """Получить метрики по классам"""
        logger.info("\n" + "="*70)
        logger.info("МЕТРИКИ ПО КЛАССАМ (ТОП-20 ПО mAP)")
        logger.info("="*70)

        if not hasattr(results, 'box'):
            logger.warning("Box metrics недоступны")
            return None

        # Собираем данные по классам
        class_data = []
        for i in range(self.num_classes):
            if i < len(results.box.p_class):
                class_data.append({
                    'class_id': i,
                    'name': self.class_names.get(i, f"Class {i}"),
                    'precision': results.box.p_class[i] if hasattr(results.box, 'p_class') else 0,
                    'recall': results.box.r_class[i] if hasattr(results.box, 'r_class') else 0,
                    'mAP50': results.box.map50_class[i] if hasattr(results.box, 'map50_class') else 0,
                    'mAP50-95': results.box.map_class[i] if hasattr(results.box, 'map_class') else 0,
                })

        if not class_data:
            logger.warning("Данные по классам недоступны")
            return None

        df = pd.DataFrame(class_data)
        df = df.sort_values('mAP50-95', ascending=False)

        # Топ 20
        for idx, row in df.head(20).iterrows():
            logger.info(
                f"  {row['class_id']:3d} {row['name']:30s} | "
                f"P:{row['precision']:.3f} R:{row['recall']:.3f} "
                f"mAP50:{row['mAP50']:.3f} mAP50-95:{row['mAP50-95']:.3f}"
            )

        logger.info("="*70)
        return df

    def save_metrics_json(self, results, df_classes=None):
        """Сохранить метрики в JSON"""
        metrics_file = self.output_dir / "metrics.json"

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": str(self.model_path),
            "conf_threshold": self.conf_threshold,
            "global_metrics": {}
        }

        if hasattr(results, 'box'):
            metrics["global_metrics"] = {
                "precision": float(results.box.p),
                "recall": float(results.box.r),
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map)
            }

        # По классам
        if df_classes is not None:
            metrics["class_metrics"] = []
            for _, row in df_classes.iterrows():
                metrics["class_metrics"].append({
                    "class_id": int(row['class_id']),
                    "name": str(row['name']),
                    "precision": float(row['precision']),
                    "recall": float(row['recall']),
                    "mAP50": float(row['mAP50']),
                    "mAP50-95": float(row['mAP50-95'])
                })

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Метрики сохранены: {metrics_file}")

    def plot_metrics(self, df_classes):
        """Построить графики метрик"""
        if df_classes is None or len(df_classes) == 0:
            logger.warning("Нет данных для графиков")
            return

        top_n = min(20, len(df_classes))
        df_top = df_classes.head(top_n)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # mAP50-95
        axes[0, 0].barh(range(len(df_top)),
                        df_top['mAP50-95'].values, color='green')
        axes[0, 0].set_yticks(range(len(df_top)))
        axes[0, 0].set_yticklabels(df_top['name'].values, fontsize=8)
        axes[0, 0].set_xlabel('mAP50-95')
        axes[0, 0].set_title('Средняя точность по классам (мАП50-95)')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Precision
        axes[0, 1].barh(range(len(df_top)),
                        df_top['precision'].values, color='blue')
        axes[0, 1].set_yticks(range(len(df_top)))
        axes[0, 1].set_yticklabels(df_top['name'].values, fontsize=8)
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_title('Точность по классам')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # Recall
        axes[1, 0].barh(range(len(df_top)),
                        df_top['recall'].values, color='orange')
        axes[1, 0].set_yticks(range(len(df_top)))
        axes[1, 0].set_yticklabels(df_top['name'].values, fontsize=8)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Полнота по классам')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # mAP50
        axes[1, 1].barh(range(len(df_top)),
                        df_top['mAP50'].values, color='red')
        axes[1, 1].set_yticks(range(len(df_top)))
        axes[1, 1].set_yticklabels(df_top['name'].values, fontsize=8)
        axes[1, 1].set_xlabel('mAP50')
        axes[1, 1].set_title('Средняя точность по классам (мАП50)')
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.suptitle('Метрики валидации по классам (топ-20)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_file = self.output_dir / "metrics_per_class.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ График сохранён: {plot_file}")

    def save_class_report(self, df_classes):
        """Сохранить отчёт по классам в CSV"""
        if df_classes is None or len(df_classes) == 0:
            return

        csv_file = self.output_dir / "class_metrics.csv"
        df_classes.to_csv(csv_file, index=False)
        logger.info(f"✓ CSV отчёт: {csv_file}")

        # Плохие классы (mAP < 0.5)
        bad_classes = df_classes[df_classes['mAP50-95'] < 0.5]
        if len(bad_classes) > 0:
            logger.warning(f"\n⚠️  КЛАССЫ С НИЗКИМ mAP50-95 (< 0.5):")
            for _, row in bad_classes.iterrows():
                logger.warning(
                    f"  {row['class_id']:3d} {row['name']:30s} "
                    f"mAP50-95: {row['mAP50-95']:.3f}"
                )

    def generate_report(self, results, df_classes):
        """Генерировать итоговый отчёт"""
        report_file = self.output_dir / "validation_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ОТЧЁТ ВАЛИДАЦИИ МОДЕЛИ YOLO\n")
            f.write("="*70 + "\n\n")

            f.write(f"Модель: {self.model_path}\n")
            f.write(f"data.yaml: {self.data_yaml}\n")
            f.write(f"Порог уверенности: {self.conf_threshold}\n")
            f.write(
                f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("ГЛОБАЛЬНЫЕ МЕТРИКИ:\n")
            f.write("-"*70 + "\n")
            if hasattr(results, 'box'):
                f.write(f"Precision: {results.box.p:.4f}\n")
                f.write(f"Recall:    {results.box.r:.4f}\n")
                f.write(f"mAP50:     {results.box.map50:.4f}\n")
                f.write(f"mAP50-95:  {results.box.map:.4f}\n\n")

            f.write(f"ВСЕГО КЛАССОВ: {self.num_classes}\n\n")

            if df_classes is not None:
                f.write("ТОП-10 ЛУЧШИХ КЛАССОВ:\n")
                f.write("-"*70 + "\n")
                for idx, row in df_classes.head(10).iterrows():
                    f.write(
                        f"{row['class_id']:3d} {row['name']:30s} "
                        f"mAP50-95: {row['mAP50-95']:.3f}\n"
                    )

                f.write("\n")
                bad = df_classes[df_classes['mAP50-95'] < 0.5]
                if len(bad) > 0:
                    f.write("КЛАССЫ С НИЗКИМ mAP50-95 (< 0.5):\n")
                    f.write("-"*70 + "\n")
                    for _, row in bad.iterrows():
                        f.write(
                            f"{row['class_id']:3d} {row['name']:30s} "
                            f"mAP50-95: {row['mAP50-95']:.3f}\n"
                        )

        logger.info(f"✓ Полный отчёт: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Расширенная валидация YOLO модели")
    parser.add_argument("model", help="Путь к best.pt")
    parser.add_argument("--data", type=str, required=True,
                        help="Путь к data.yaml")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Порог уверенности")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU ID или 'cpu'")
    parser.add_argument("--output", type=str, default="validation_results",
                        help="Директория для результатов")

    args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(
            args.model,
            args.data,
            conf_threshold=args.conf,
            output_dir=args.output
        )

        # Валидация
        results = evaluator.validate(device=args.device)

        # Метрики
        evaluator.print_global_metrics(results)
        df_classes = evaluator.get_per_class_metrics(results)

        # Сохранение результатов
        evaluator.save_metrics_json(results, df_classes)
        if df_classes is not None:
            evaluator.plot_metrics(df_classes)
            evaluator.save_class_report(df_classes)
            evaluator.generate_report(results, df_classes)

        logger.info("\n✓ Валидация завершена успешно!")
        logger.info(f"✓ Результаты сохранены в: {evaluator.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
