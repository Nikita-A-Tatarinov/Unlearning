import ast
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from SafeRLHF.config import (ORIGINAL_DATA_FILE, CLS_DATA_DIR,
                             CLS_SAFE_DATA_FILE_NAME, CLS_SAFE_DATA_FILE,
                             CLS_TRAIN_TEST_DATA_DIR, CLS_DATA_STATS_FILE)

from SafeRLHF.processing.config import TEST_RATIO


def create_empty_file(file_path: str, columns: list[str]) -> None:
    df = pd.DataFrame(columns=columns)
    df.to_csv(file_path, index=False)


def output_train_test(df: pd.DataFrame, output_dir: str, file_name: str, test_ratio: float) -> None:
    train_df, test_df = train_test_split(df, test_size=test_ratio,
                                         random_state=42)
    train_df.to_csv(os.path.join(output_dir, f'{file_name}_train.csv'),
                    mode='a', header=False, index=False)
    test_df.to_csv(os.path.join(output_dir, f'{file_name}_test.csv'),
                   mode='a', header=False, index=False)


def save_safe_data(df: pd.DataFrame, file_path: str, stats_file: str) -> None:
    safe_df = pd.concat([(df[df['is_response_0_safe']][['prompt', 'response_0']]
                        .rename(columns={'response_0': 'response'})),
                        (df[df['is_response_1_safe']][['prompt', 'response_1']]
                        .rename(columns={'response_1': 'response'}))
                         ])
    safe_df.dropna(inplace=True)
    safe_df.to_csv(file_path, index=False)
    pd.DataFrame([[os.path.basename(file_path)[:-4], len(safe_df)]],
                 columns=['category', "num_observations"]).to_csv(
                     stats_file, mode='a', header=False, index=False)
    return safe_df


def convert_category_strings_to_dicts(df: pd.DataFrame) -> None:
    for i in range(2):
        df[f'response_{i}_harm_category'] = df[f'response_{i}_harm_category'].apply(
            ast.literal_eval)


def create_safety_categories_files(categories: list,
                                   output_dir: str,
                                   train_test_dir: str) -> None:
    '''
    Creates empty files for each category to fill later.
    '''
    for cat in categories:
        create_empty_file(os.path.join(output_dir, f'{cat}.csv'),
                          ['prompt', 'response'])
        create_empty_file(os.path.join(train_test_dir, f'{cat}_train.csv'),
                          ['prompt', 'response'])
        create_empty_file(os.path.join(train_test_dir, f'{cat}_test.csv'),
                          ['prompt', 'response'])


def save_unsafe_data_by_category(df: pd.DataFrame,
                                 i: int,
                                 categories: list,
                                 output_dir: str,
                                 train_test_output_dir: str,
                                 stats_file: str,
                                 test_ratio: float) -> None:
    unsafe_df = df[~df[f'is_response_{i}_safe']]
    # Making column of boolean values for each category
    # and concatenating them with prompt-response pairs
    unsafe_df = pd.concat([(unsafe_df[['prompt', f'response_{i}']]
                            .rename(columns={f'response_{i}': 'response'})
                            .reset_index(drop=True)),
                           (pd.json_normalize(unsafe_df[f'response_{i}_harm_category'])
                            .reset_index(drop=True))
                           ], axis=1)

    for cat in categories:
        unsafe_cat_df = (unsafe_df[unsafe_df[cat].astype(bool)]
                         [['prompt', 'response']].dropna()
                         .reset_index(drop=True))
        unsafe_cat_df.to_csv(os.path.join(output_dir, f'{cat}.csv'),
                             mode='a', header=False, index=False)
        pd.DataFrame([[cat, len(unsafe_cat_df)]],
                     columns=['category', "num_observations"]).to_csv(
            stats_file, mode='a', header=False, index=False)
        output_train_test(
            unsafe_cat_df, train_test_output_dir, cat, test_ratio)


def main():
    create_empty_file(CLS_DATA_STATS_FILE, ['category', 'num_observations'])
    original_df = pd.read_csv(ORIGINAL_DATA_FILE)

    safe_df = save_safe_data(
        original_df, CLS_SAFE_DATA_FILE, CLS_DATA_STATS_FILE)
    create_empty_file(os.path.join(CLS_TRAIN_TEST_DATA_DIR,
                                   f'{CLS_SAFE_DATA_FILE_NAME}_train.csv'),
                      ['prompt', 'response'])
    create_empty_file(os.path.join(CLS_TRAIN_TEST_DATA_DIR,
                                   f'{CLS_SAFE_DATA_FILE_NAME}_test.csv'),
                      ['prompt', 'response'])
    output_train_test(safe_df, CLS_TRAIN_TEST_DATA_DIR,
                      CLS_SAFE_DATA_FILE_NAME, TEST_RATIO)

    convert_category_strings_to_dicts(original_df)
    safety_categories = list(
        original_df['response_0_harm_category'].iloc[0].keys())
    create_safety_categories_files(safety_categories, CLS_DATA_DIR,
                                   CLS_TRAIN_TEST_DATA_DIR)

    for i in range(2):
        save_unsafe_data_by_category(
            original_df, i, safety_categories, CLS_DATA_DIR,
            CLS_TRAIN_TEST_DATA_DIR, CLS_DATA_STATS_FILE,
            TEST_RATIO)


if __name__ == '__main__':
    main()
