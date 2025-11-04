def create_super_categories(df):
    category_mapping = {}
    categories = df['good'].unique()

    for category in categories:
        category_lower = str(category).lower()

        # Одежда
        if any(word in category_lower for word in
               ['брюки', 'футболка', 'платье', 'кофта', 'рубашка', 'юбка', 'куртка', 'пальто', 'джемпер', 'шорты',
                'трусы', 'носки', 'кроссовки', 'белье', 'свитер', 'блузка', 'пиджак', 'жилет', 'майка', 'бант',
                'заколка', 'пуловер', 'блуза', 'фуфайка', 'сандал', 'одежда', 'кардиган', 'купальник']):
            category_mapping[category] = 'одежда'
        # Еда
        elif any(word in category_lower for word in
                 ['пиво', 'вода', 'молоко', 'хлеб', 'печенье', 'сыр', 'колбаса', 'сок', 'чай', 'масло', 'напиток',
                  'шоколад', 'мороженое', 'йогурт', 'корм', 'салат', 'конфеты', 'кофе', 'сахар', 'мука', 'консерв',
                  'колбас', 'мясо', 'рыба', 'овощ', 'фрукт', 'банан', 'яблоко', 'макарон', 'хлопь', 'коктейль', 'квас',
                  'рулет', 'круп', 'пицц', 'пельмен', 'драже', 'нектар', 'пирог', 'пломбир', 'перец', 'семен', 'огурц',
                  'булочк', 'десерт', 'сироп', 'икра', 'соль', 'капуста', 'батон']):
            category_mapping[category] = 'еда'
        # Электроника
        elif any(word in category_lower for word in
                 ['батарейк', 'аккумулятор', 'зарядк', 'кабель', 'провод', 'розетк', 'выключатель']):
            category_mapping[category] = 'электроника'
        # Канцелярия
        elif any(word in category_lower for word in
                 ['книга', 'учебник', 'тетрадь', 'блокнот', 'дневник', 'альбом', 'ручка', 'карандаш', 'бумага', 'краск',
                  'кисть', 'канцеляр', 'конструктор', 'маркер', 'резинка']):
            category_mapping[category] = 'канцелярия'
        # Стройматериалы
        elif any(word in category_lower for word in
                 ['труба', 'лак', 'краск', 'эмаль', 'герметик', 'пена', 'валик', 'угол', 'кран', 'шланг', 'смеситель']):
            category_mapping[category] = 'стройматериалы'
        # Здоровье
        elif any(word in category_lower for word in
                 ['таблетки', 'лекарств', 'витамины', 'медицин', 'аптечка', 'препарат', 'болеутол', 'здоровье',
                  'медицинск', 'шприц', 'лейкопластырь', 'прокладк', 'презерватив']):
            category_mapping[category] = 'здоровье'
        # Косметика
        elif any(word in category_lower for word in
                 ['крем', 'шампунь', 'косметика', 'мыло', 'гель', 'лосьон', 'дезодорант', 'парфюм', 'духи', 'туалетн',
                  'гигиен', 'жидкост', 'салфетк', 'ароматизатор', 'кондиционер']):
            category_mapping[category] = 'косметика'
        # Дом
        elif any(word in category_lower for word in
                 ['мебель', 'посуда', 'ковер', 'шторы', 'бытов', 'лампа', 'светильник', 'интерьер', 'кухон', 'ванн',
                  'хозтовар', 'чехол', 'кашпо']):
            category_mapping[category] = 'дом'
        # Табак
        elif any(word in category_lower for word in ['сигарет', 'табак', 'никотин']):
            category_mapping[category] = 'табак'
        # Другое
        elif df[df['good'] == category].shape[0] > 25:
            category_mapping[category] = category
        else:
            category_mapping[category] = 'другое'

    df['super_category'] = df['good'].map(category_mapping)
    return df


def analyze_other_category(df):
    other_df = df[df['super_category'] == 'другое']

    print("\n" + "=" * 100)
    print("Анализ категории 'другое'")

    other_categories = other_df['good'].value_counts().head(20)
    print("Топ-20 категорий в 'другое':")
    for category, count in other_categories.items():
        print(f"  -{category}: {count} samples")

    from collections import Counter
    all_words = []
    for name in other_df['name'].dropna():
        words = str(name).lower().split()
        all_words.extend(words)

    common_words = Counter(all_words).most_common(15)
    print(f"\nЧастые слова в 'другое':")
    for word, count in common_words:
        print(f"  -{word}: {count}")

    return other_df


def test_super_categories():
    from explore import analyze_data

    df = analyze_data()
    if df is not None:
        df_processed = create_super_categories(df)
        print(f"\nУникальных супер-категорий: {df_processed['super_category'].nunique()}")
        print("Распределение супер-категорий:")
        print(df_processed['super_category'].value_counts())

        analyze_other_category(df_processed)

        return df_processed
    return None


if __name__ == "__main__":
    test_super_categories()
