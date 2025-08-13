"""
ModeloBase para a tarefa de previsão
"""

from catboost import CatBoostClassifier


model = CatBoostClassifier(
    auto_class_weights = 'Balanced',
    eval_metric = 'AUC',
    loss_function = 'Logloss',
)