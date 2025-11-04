import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import os

from src.data.preprocess import create_super_categories


def create_baseline_model():
    print("\n" + "=" * 100)
    print("Создание базовой модели")

    data_path = '../../data/raw/train_supervised_dataset.csv'
    df = pd.read_csv(data_path)
    df_processed = create_super_categories(df)

    print(f"Данные для обучения: {df_processed.shape}")
    print(f"Целевые категории: {df_processed['super_category'].nunique()}")

    X = df_processed['name']
    y = df_processed['super_category']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Обучающая выборка: {x_train.shape[0]} samples")
    print(f"Тестовая выборка: {x_test.shape[0]} samples")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    print(f"   Размерность признаков: {x_train_tfidf.shape}")
    print(f"   Размер словаря: {len(vectorizer.vocabulary_)}")

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    classifier.fit(x_train_tfidf, y_train)
    print(f"   Количество классов: {len(classifier.classes_)}")
    y_pred = classifier.predict(x_test_tfidf)

    print("\n" + "=" * 100)
    print("\nРезультаты")
    print(f"F1-score (micro): {f1_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-score (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    print("\n" + "=" * 100)
    print("\nОтчет по классификации (топ-10 категорий):")
    top_categories = y_test.value_counts().head(10).index
    y_test_top = y_test[y_test.isin(top_categories)]
    y_pred_top = y_pred[pd.Series(y_test).isin(top_categories)]

    print(classification_report(y_test_top, y_pred_top, zero_division=0))

    os.makedirs('../../models', exist_ok=True)

    joblib.dump(vectorizer, '../../models/vectorizer.joblib')
    joblib.dump(classifier, '../../models/classifier.joblib')

    return classifier, vectorizer, x_test, y_test, y_pred


def load_model():
    vectorizer = joblib.load('../../models/vectorizer.joblib')
    classifier = joblib.load('../../models/classifier.joblib')
    return classifier, vectorizer


def predict_new_text(texts, classifier, vectorizer):
    texts_tfidf = vectorizer.transform(texts)
    predictions = classifier.predict(texts_tfidf)
    return predictions


def create_submission():
    classifier, vectorizer = load_model()

    test_df = pd.read_csv('../../data/raw/test_dataset.csv')
    predictions = predict_new_text(test_df['name'], classifier, vectorizer)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'good': predictions,
        'brand': ''
    })

    submission.to_csv('../../submission.csv', index=False)

    return submission


if __name__ == "__main__":
    create_baseline_model()
    create_submission()