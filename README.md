# Gradient-boosting-from-scratch

В данном проекте представлена кастомная реализация градипентного бустинга с нуля с его применением на датасете.

Цель - разобраться с внутренним устройством классического бустинга и его имплементаций и сравнить свою реализацию с версиями бустинга из различных библиотек таких как LightGBM, XGBoost, CatBoost.

---

## Data

В проекте используется датасет о доходах - Adult Income dataset (UCI)

Задача: бинарная классификация - предсказать превышает ли доход индивида 50к$ в год.

---

## Project structure

Проект состоит из 4 частей:

1. **EDA**
   - Анализ распределений признаков
   - Изучение взаимосвязей между признаками и целевой переменной
   - Поиск нелинейных взаимосвязей

2. **Baseline models**
   - Логистическая регрессия из sklearn 
   - Random Forest из sklearn
   - Кастомный градиентный бустинг

3. **Ablation study**
   - Добавление и анализ различных имплементаций градиентного бустинга:
     - Bootstrap (Bernoulli, Bayesian)
     - Feature subsampling (RSM)
     - Quantization
     - GOSS
     - DART

4. **Benchmark vs libraries**
   - Сравнение с библиотеками:
     - LightGBM
     - XGBoost
     - CatBoost

---

## Custom Gradient Boosting

Модель реализована в файле `boosting.py`.

Ключевые характеристики:

- Логистическая функция потерь для бинарной классификации
- Кодирование целевой переменной на отрезке {-1, 1}
- Решающие деревья небольшой глубины как базовые модели
- Ранняя остановка (early stopping)
- Feature subsampling (RSM)
- Bootstrap (Bernoulli / Bayesian)
- Quantization (binning)
- GOSS (Gradient-based sampling)
- DART (dropout boosting)

---

## Results

All models achieve similar ROC-AUC on the Adult dataset:

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | ~0.90 |
| Random Forest | ~0.89 |
| Custom GBM | ~0.92 |
| LightGBM | ~0.92 |
| XGBoost | ~0.92 |
| CatBoost | ~0.92 |

---

## Key Insights

- В данных есть достаточно сильный, но в то же время простой сигнал 
- Линейные модели достаточно хорошо справляются с задачей после простого OHE категориальных признаков
- Улучшения бустинга дали небольшой прирост в качестве
- Кастомная реализация бустинга выбивает схожее качество на тесте, не уступая реализациям из библиотек


---

## How to run

```bash
pip install -r requirements.txt
