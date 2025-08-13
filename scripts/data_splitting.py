"""
Script para a divisão estratificada dos dados em treino, validação e teste

As funções do script permitem:
--- Carregar o dataset processado
--- Fazer o split do dataset
--- Identificar as features categóricas
--- Guardar os splits e as outras informações para serem utilizadas posteriormente

Autor: Carlos Gomes
"""

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import pickle
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:

    """
    Classe para dividir dados em treino, validação e teste de forma estratificada
    """

    def __init__(self, test_size=0.2, validation_size=0.2, random_state=None):

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state


    @staticmethod
    def load_dataset(dataset_path):

        """
        Carrega o dataset a partir do path

        Arguments:
            dataset_path (str/Path): Caminho para o ficheiro

        Returns:
            pd.DataFrame: Dataset carregado
        """

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {dataset_path}")

        if dataset_path.suffix == '.csv':
            dataset = pd.read_csv(dataset_path)
            logger.info(f"Dataset carregado do csv: {dataset_path}")
        elif dataset_path.suffix == '.xlsx':
            dataset = pd.read_excel(dataset_path)
            logger.info(f"Dataset carregado do xlsx: {dataset_path}")
        else:
            raise ValueError(f"Formato do ficheiro não é compatível com o workflow -> Aceita csv e xlsx")

        return dataset



    def split_data(self, dataset, target_col='Elegível?'):

        """
        Divide o dataset em treino, validação e teste com estratificação

        Arguments:
            dataset (pd.DataFrame): Dataset completo
            target_col (str): Nome da coluna target

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """

        x = dataset.drop(target_col, axis=1)
        y = dataset[target_col]

        logger.info(f"Dataset original: {x.shape[0]} linhas, {x.shape[1]} features")
        logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, stratify=y,
                                                          random_state=self.random_state, shuffle=True)

        val_size_adjusted = self.validation_size / (1 - self.test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size_adjusted,
                                                            stratify=y_train, random_state=self.random_state,
                                                            shuffle=True)

        logger.info(f"Treino: {x_train.shape} ({len(y_train[y_train == 1])} elegíveis)")
        logger.info(f"Validação: {x_val.shape} ({len(y_val[y_val == 1])} elegíveis)")
        logger.info(f"Teste: {x_test.shape} ({len(y_test[y_test == 1])} elegíveis)")

        return x_train, x_val, x_test, y_train, y_val, y_test


    @staticmethod
    def identify_categorical_features(x_train):

        """
        Identifica as colunas categóricas

        Arguments:
            x_train (pd.DataFrame): Dataset de treino

        Returns:
            list: Lista de nomes das colunas categóricas
        """

        categorical_cols = [
            col for col in x_train.columns
            if x_train[col].dtype == 'object' or str(x_train[col].dtype) == 'category'
        ]

        logger.info(f"Colunas categóricas identificadas: {categorical_cols}")
        return categorical_cols


    def save_splits(self, splits_data, output_dir, random_state=None):

        """
        Guardar as divisões dos dados

        Arguments:
            splits_data (dict): Dicionário com os dados divididos
            output_dir (str): Local onde guardar os ficheiros
            random_state (int): Número utilizado para guardar as características do split
        """

        script_dir = Path(__file__).parent
        project_root = script_dir.parent

        if isinstance(output_dir, str):
            full_output_path = project_root / output_dir
        else:
            full_output_path = Path(output_dir)

        if random_state is not None:
            full_output_path = full_output_path / f"split_{random_state}"
            logger.info(f"A criar subpasta para random_state={random_state}")

        full_output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"A guardar splits em: {full_output_path}")

        for name, data in splits_data.items():
            file_path = full_output_path / (name + '.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Guardado: {file_path}")



def main():

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    dataset_path = project_root / 'data' / 'processed' / 'Ficheiro_Compras_Processado.csv'
    output_dir = project_root / 'outputs' / 'splits'

    try:
        random_state = 1

        spliter = DataSplitter(test_size=0.2, validation_size=0.2, random_state=random_state)

        dataset = spliter.load_dataset(dataset_path)

        x_train, x_val, x_test, y_train, y_val, y_test = spliter.split_data(dataset)

        categorical_cols = spliter.identify_categorical_features(x_train)

        splits_data = {"x_train": x_train,
                       "x_val": x_val,
                       "x_test": x_test,
                       "y_train": y_train,
                       "y_val": y_val,
                       "y_test": y_test,
                       "categorical_cols": categorical_cols
                       }

        spliter.save_splits(splits_data, output_dir, random_state=random_state)
        final_output_path = script_dir.parent / output_dir / f"split_{random_state}"

        print("\n" + "=" * 60)
        print("Divisão dos dados:")
        print("=" * 60)
        print(f"Treino:     {x_train.shape[0]:,} linhas ({len(y_train[y_train == 1]):,} elegíveis)")
        print(f"Validação:  {x_val.shape[0]:,} linhas ({len(y_val[y_val == 1]):,} elegíveis)")
        print(f"Teste:      {x_test.shape[0]:,} linhas ({len(y_test[y_test == 1]):,} elegíveis)")
        print(f"Ficheiros guardados em: {final_output_path}")

        logger.info("Divisão dos dados concluída com sucesso")

    except Exception as e:

        logger.error(f"Erro durante a execução: {e}")
        raise

if __name__ == '__main__':
    main()