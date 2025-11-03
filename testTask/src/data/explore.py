import pandas as pd


def analyze_data():
    data_path = '../../data/raw/train_supervised_dataset.csv'

    try:
        train_df = pd.read_csv(data_path)

        print(f"Размер данных: {train_df.shape}")
        print(f"Колонки: {list(train_df.columns)}\n")

        print(f"Уникальных категорий: {train_df['good'].nunique()}")
        print("Топ-5 категорий:")
        print(train_df['good'].value_counts().head())

        print("\nПримеры данных:")
        print(train_df[['name', 'good', 'brand']].head(5))

        print("=" * 100)

        category_counts = train_df['good'].value_counts()
        print(f"   -Самая частая категория: '{category_counts.index[0]}' ({category_counts.iloc[0]} samples)")
        print(f"   -Самая редкая категория: '{category_counts.index[-1]}' ({category_counts.iloc[-1]} samples)")
        print(f"   -Медианное количество: '{category_counts.median()}'")

        print(f"   -Заполнено брендов: {train_df['brand'].notna().sum()} из {len(train_df)}")
        print(f"   -Уникальных брендов: {train_df['brand'].nunique()}")

        train_df['name_length'] = train_df['name'].str.len()
        print(f"   -Мин. длина названия: {train_df['name_length'].min()} симв.")
        print(f"   -Макс. длина названия: {train_df['name_length'].max()} симв.")
        print(f"   -Средняя длина: {train_df['name_length'].mean():.1f} симв.")

        return train_df

    except FileNotFoundError:
        print("Файл не найден!")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


if __name__ == "__main__":
    analyze_data()
