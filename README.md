# Оценка А/Б теста и кластеризация для магазина спортивных товаров
https://github.com/Dimayo/sporting_store/blob/main/research.ipynb <br>
Библиотеки python: pandas, sklearn, matplotlib, sqlalchemy, scipy, seaborn, numpy, re, kmodes

## Цель проекта
C помощью данных о покупках клиентов и их социально-демографических признаках проанализировать эффективность уже проведённых маркетинговых кампаний и выявить факторы, способные повысить продажи.

А именно: 
* Провести расчёт A/B-теста и посчитать значения основных метрик. Потом сделать бизнес-рекомендацию и обосновать её. 
* Выяснить, на какие кластеры разбивается аудитория, и предложить методы работы с каждым кластером.
* Построить модель склонности клиента к покупке определённого товара при коммуникации, основанную на данных о профилях клиентов, данных товаров и данных о прошлых маркетинговых кампаниях.

## Описание проекта
База данных содержит три таблицы:

* personal_data — ID клиентов, их пол, возраст, образование, страна и город проживания;
* personal_data_coeffs — данные с персональными коэффициентами клиентов, которые рассчитываются по некоторой закрытой схеме (вам потребуется коэффициент personal_coef); 
* purchases — данные о покупках: ID покупателя, название товара, цвет, стоимость, гендерная принадлежность потенциальных покупателей товара, наличие скидки (поле base_sale. Значение 1 соответствует наличию скидки на момент покупки) и дата покупки. 


При передаче данных выяснилось, что часть информации о клиентах из таблицы personal_data была утеряна. Поэтому, помимо базы данных, предоставлен сжатый CSV-файл с утерянными данными (personal_data.csv.gz). К сожалению, информацию о поле клиентов восстановить не удалось.

Известно, что магазин проводил две маркетинговые кампании: 

* Первая кампания проводилась в период с 5-го по 16-й день, ID участвовавших в ней пользователей содержатся в файле ids_first_company_positive.txt. Эта кампания включала в себя предоставление персональной скидки 5 000 клиентов через email-рассылку.
* Вторая кампания проводилась на жителях города 1 134 и представляла собой баннерную рекламу на билбордах: скидка всем каждое 15-е число месяца (15-й и 45-й день в нашем случае).

## Что было сделано
### Подготовка данных
Было произведено подключение к базе данных:
```
engine = create_engine('sqlite:///data/shop_database.db')
conn = engine.connect()
```
Загружены таблицы из базы:
```
df_pers = pd.read_sql(sql=text('SELECT * FROM personal_data'),con=conn)
```
Загружены утерянные данные:
```
df_lost = pd.read_csv('data\personal_data.csv.gz', compression='gzip', header=0)
```
Данные о клиентах объединены с утерянными данными:
```
df_clients = pd.concat([df_pers, df_lost])
```
Созданы визуализации нескольких распределений из которых можно сделать вывод о том, что распределение возраста клиентов близко к нормальному:

<img src="https://github.com/Dimayo/sporting_store/assets/44707838/d825e0ff-7644-4410-b0d1-2ead3882d592" width="600"> <br> <br>
Клиенты преимущественно мужского пола:

<img src="https://github.com/Dimayo/sporting_store/assets/44707838/4202686d-3197-49fb-b370-0b1cbb81df4b" width="600"> <br> <br>

Клиенты магазина имеют преимущественно среднее образование:

<img src="https://github.com/Dimayo/sporting_store/assets/44707838/d021fa1b-d995-4328-8c73-3b04388504c1" width="600"> <br> <br>
Так как признак пол в файле с утерянными данными не был заполнен, он был заполнен с помощью модели градиентного бустинга после предварительного подбора гиперпараметров:
```
model = GradientBoostingClassifier(subsample=0.4, n_estimators=300, min_samples_split=16,
                                   min_samples_leaf=2, max_features='sqrt', max_depth=34,
                                   loss= 'exponential', learning_rate=1,
                                   criterion='squared_error').fit(x_train, y_train)
df_test['gender'] = model.predict(x_test)
```
В таблице с данными о покупках также были заполнены пропуски и обработаны текстовые признаки:
```
df_purch['product'] = df_purch['product'].apply(lambda x: " "
                                                .join(re.findall('[А-Яа-я]{3,20}', x)).lower())
df_purch['colour'] = df_purch['colour'].apply(lambda x: x.split('/')[0].lower())
```
### Оценка А/Б теста
Загружен файл, который содержит id клиентов учавствовавших в А/Б тесте:
```
with open('data/ids_first_company_positive.txt') as f:
    positive = f.read()
```
Файл был обработан для получения только необходимых данных:
```
elem_list = re.split(';|,| ', positive)

positive_id = []

for i in elem_list:
    if i.isdigit():
        positive_id.append(int(i))
```
Такие же действия были произведены с файлом, который содержит id клиентов из контрольной группы. Далее была произведена группировка по id, чтобы получить выручку на одного клиента:
```
arpu_group = df_test.groupby('id', as_index=False).agg({'cost':'sum'})
```
И созданы контрольная и тестовая группа:
```
positive_arpu = arpu_group.loc[arpu_group['id'].isin(positive_id), :]
negative_arpu = arpu_group.loc[arpu_group['id'].isin(negative_id), :]

```
Группы были проверены на несоответствие коэффициента выборки (SRM):
```
test = len(positive_arpu)
control = len(negative_arpu)
overall = test + control

observed = [test, control]
expected = [overall / 2, overall / 2]

chi = stats.chisquare(observed, f_exp=expected)
print(chi)

if chi[1] < 0.01:
    print('SRM присутствует')
else:
    print('SRM отсутсвует')

```
Было протестировано несколько гипотез, нормальность распределения проверялась с помощью теста Шапиро-Уилка или Колмогорова-Смирнова. Так как распределения не являются нормальными, а выборки независимы, для проверки гипотезы о среднем использовался критерий Манна-Уитни:
```
mann = stats.mannwhitneyu(positive_arpu['cost'], negative_arpu['cost'])
print(mann)

if mann[1] < 0.05:
    print('Есть статистически значимая разница')
else:
    print('Статистически значимой разницы нет')

```
### Кластеризация
Создана новая таблица с группировкой по id:
```
df_g = df_purch.groupby('id', as_index=False).agg({'cost':'sum',
                                                   'product': pd.Series.mode,
                                                   'colour': pd.Series.mode,
                                                   'product_sex': pd.Series.mode,
                                                   'base_sale':'mean',
                                                   'dt':'max'
                                                   })

```

Изменены названия признаков:
```
dict = {'cost': 'cost_sum',
        'product': 'product_mode',
        'colour': 'color_mode',
        'product_sex': 'product_sex_mode',
        'base_sale': 'best_sale_mean',
        'dt': 'dt_max'}

df_g = df_g.rename(columns=dict)
```
Преобразованы элементы признаков, которые состоят из списков:
```
mode_columns = ['product_mode', 'color_mode', 'product_sex_mode']

for column in mode_columns:
    df_g[column] = df_g[column].apply(lambda x: x[0] if type(x) == np.ndarray else x)
```
Датафрейм объединен с данными о клиентах:
```
df = pd.merge(left=df_clients, right=df_g, on='id', how='inner')
```
Числовые признаки нормализованы с помощью Standart Scaler:
```
num_columns = ['age', 'personal_coef','cost_sum', 'best_sale_mean','dt_max']
scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])
```
Для кластеризации выбран алгоритм Kprototypes, так как датафрейм содержит категориальные данные. Оптимальное число кластеров найдено с помощью метода "локтя":
```
n_clusters = list(range(2,8))
cost = []

for n in n_clusters:
    kproto = KPrototypes(n_clusters=n, init='Cao', n_jobs=-1)
    kproto.fit_predict(df, categorical=[0,2,3,4,7,8,9])
    cost.append(kproto.cost_)
```
<img src="https://github.com/Dimayo/sporting_store/assets/44707838/48e7bb24-c836-4203-b382-096f3e5b18e1" width="600"> <br> <br>

### Модель склонности клиента к покупке определенного товара
Была создана таблица с id клиентов, участвовавшими в первой и второй маркетинговых компаниях и продуктами, которые они покупали:
```
df_first = df_purch[(df_purch['dt'] >= 5) & (df_purch['dt'] < 17)]
product_group = df_first.groupby('id', as_index=False).agg({'product': pd.Series.mode})
first_camp = product_group.loc[product_group['id'].isin(positive_id), :]
first_camp['product'] = first_camp['product'].apply(lambda x: x[0] if type(x) == np.ndarray else x)

df_second = df_purch[(df_purch['dt'] == 15) | (df_purch['dt'] == 45)]
second_camp = df_second.groupby('id', as_index=False).agg({'product': pd.Series.mode})
second_camp['product'] = second_camp['product'].apply(lambda x: x[0] if type(x) == np.ndarray else x)
```
Далее таблицы о купленных продуктах были объединены с информацией о клиентах:
```
first_df = pd.merge(left=df_clients, right = first_camp, on='id', how='inner')
first_df.shape

second_df = pd.merge(left=df_clients, right = second_camp, on='id', how='inner')
second_df = second_df[second_df.city == 1134]
second_df.shape
```
Датафреймы с информацией о клиентах и купленных товарах в первой и второй маркетинговых компаниях были объединены между собой:
```
train = first_df.append(second_df)
```
Были удалены дубликаты и упрощено название товаров для сокращения количества категорий:
```
train['product'] = train['product'].apply(lambda x: x.split(' ')[0])
```
Создан тестовый датафрейм на котором необходимо применить модель:
```
test = df_clients[(df_clients.city == 1188) & (df_clients.country == 32)]
```
Далее был осуществлен подбор гиперпараметров модели с помощью Randomized Search, и обучена модель показавшая лучшие результаты:
```
params = {'n_estimators' : [300, 500, 700],
          'max_depth': np.arange(10, 60, 4),
          'min_samples_leaf': np.arange(1, 10, 1),
          'min_samples_split': np.arange(2, 20, 2),
          'class_weight': ('balanced', None)}

rs = RandomizedSearchCV(rfc, params, cv=kf, scoring='f1_micro', n_jobs=-1, error_score='raise')
rs.fit(x_train, y_train)

print('Best params: ', rs.best_params_)
print('Best score: ', rs.best_score_)
```
Сделано предсказание на тестовой выборке и создан новый признак product:
```
model_pred = model.predict(x_test)
x_test['product'] = model_pred
```

## Результат
### Оценка А/Б теста

