"""
Script para realizar o load dos dados de treino, validação e teste (Pode-se escolher conforme o random_state)

As funções do script permitem:
--- Dar load aos dados conforme as divisões aplicadas previamente

Autor: Carlos Gomes
"""

from pathlib import Path
import pickle
import logging


logger = logging.getLogger(__name__)

def load_data_splits(splits_path = "outputs/splits", random_state=None):

    """
     Carregar os splits dos dados previamente guardados

     Arguments:
         splits_path (str): Caminho para os ficheiros
         random_state (int): Random state usado na divisão (para encontrar subpasta split_X)

     Returns:
         dict: Dicionário com todos os dados
     """

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if isinstance(splits_path, str):
        full_splits_path = project_root / splits_path
    else:
        full_splits_path = Path(splits_path)

    if random_state is not None:
        full_splits_path = full_splits_path / f"split_{random_state}"
        logger.info(f"A carregar dados do split_{random_state}")

    required_files = ['x_train.pkl',
                      'x_val.pkl',
                      'x_test.pkl',
                      'y_train.pkl',
                      'y_val.pkl',
                      'y_test.pkl',
                      'categorical_cols.pkl'
                      ]

    for file in required_files:
        file_path = full_splits_path / file
        if not file_path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")

    loaded_data = {}
    for file_name in required_files:
        name = file_name.replace('.pkl', '')
        file_path = full_splits_path / file_name

        with open(file_path, 'rb') as f:
            loaded_data[name] = pickle.load(f)

        logger.info(f"Carregado: {file_name}")

    return loaded_data


if __name__ == "__main__":

    try:
        data = load_data_splits(random_state=1)
        print("Dados carregados com sucesso!")
        print(f"Validação: {data['x_val'].shape}")
        print(f"Teste: {data['x_test'].shape}")
        print(f"Categóricas: {data['categorical_cols']}")

    except Exception as e:
        print(f"Erro ao carregar dados: {e}")