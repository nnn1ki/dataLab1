# импорт пакетов
import inline
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

names = {'Serial Number', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases', 'Total Test', 'Population'}

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

# чтение данных
df = pd.read_csv(r'C:\Users\nma1t\Downloads\archive (1)\covid_worldwide.csv')
df.fillna(0)

def makeNum(name):
    df[name] = df[name].replace(',', '', regex=True)
    df[name] = df[name].astype(float)

def inferValues(df):
    df = df.infer_objects()

def makeStr(name):
    df[name] = df[name].astype(str)

def makeHeatChart():
    cols = df.columns[:8]  # первые 30 колонок
    # желтый - пропущенные данные, синий - не пропущенные
    colours = ['#000099', '#ffff00']
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
    pl.show()  # показываем тепловой график

def percentageSkipping():
    print("")
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))

def makeHistogram():
    # сначала создаем индикатор для признаков с пропущенными данными
    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            print('created missing indicator for: {}'.format(col))
            df['{}_ismissing'.format(col)] = missing

    # затем на основе индикатора строим гистограмму
    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)

    df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
    pl.show()

def makeBoxHistogram(name):
    df.boxplot(column=[name])
    pl.show()

def discardingOmissions():
    # отбрасывание с большим количеством пропущенных значений
    print('\n' + "отбрасывание с большим количеством пропусков")
    ind_missing = df[df['num_missing'] > 35].index
    df_less_missing_rows = df.drop(ind_missing, axis=0)
    print(df_less_missing_rows)

def rejectionAttribute(name):
    cols_to_drop = [name]
    df_less_hos_beds_raion = df.drop(cols_to_drop, axis=1)
    print(df_less_hos_beds_raion)

def addValues():
    # заполнение недостающих значений медианой
    # только для числовых значений
    # impute the missing values and create the missing value indicator variables for each numeric column.
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values

    for col in numeric_cols:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:  # only do the imputation for the columns that have missing values.
            print('imputing missing values for: {}'.format(col))
            df['{}_ismissing'.format(col)] = missing
            med = df[col].median()
            df[col] = df[col].fillna(med)
            #почему-то цвета становятся инверсными
    percentageSkipping()

def makeDescriptiveStatistics(name):
    print(df[name].describe())
    print("")

def makeBarChart(name):
    df[name].value_counts().plot.bar()
    plt.show()

def mostIsSame():
    num_rows = len(df.index)
    low_information_cols = []

    for col in df.columns:
        cnts = df[col].value_counts(dropna=False)
        top_pct = (cnts / num_rows).iloc[0]

        if top_pct > 0.95:
            low_information_cols.append(col)
            print('{0}: {1:.5f}%'.format(col, top_pct * 100))
            print(cnts)
            print()


def findSameBySerialNumber(name):
    # отбрасываем неуникальные строки
    df_dedupped = df.drop(name, axis=1).drop_duplicates()

    # сравниваем формы старого и нового наборов
    print('\n' + "Результаты отбрасывания дубликатов")
    print(df.shape)
    print(df_dedupped.shape)

def findSameByCriticalParametersAndDelete():
    key = ['Country', 'Total Cases', 'Total Deaths', 'Total Test']
    print(df.fillna(-999).groupby(key)['Serial Number'].count().sort_values(ascending=False).head(20))

    df_dedupped2 = df.drop_duplicates(subset=key)

    print(df.shape)
    print(df_dedupped2.shape)



#todo
#убрать выбрасы
#диаграммы клилендла





print('\n' + "До преобразования")
# shape and data types of the data
print(df.shape)
print(df.dtypes)

# прелбразование типа
makeStr('Country')
makeNum('Total Cases')
makeNum('Total Recovered')
makeNum('Total Deaths')
makeNum('Active Cases')
makeNum('Total Test')
makeNum('Population')
makeNum('Total Cases')
# inferValues(df)


print('\n' + "После преобразования")
print(df.shape)
print(df.dtypes)

# отбор числовых колонок
print('\n' + "Отбор числовых колонок")
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# отбор нечисловых колонок
print('\n' + "Отбор не числовых колонок")
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

# тепловая карта для визуализации пропусков
makeHeatChart()

#процентный список пропущенных данных
percentageSkipping()

# Гистограмма пропущенных данных
makeHistogram()

# отбрасываем строки с большим количеством пропусков
discardingOmissions()

# отбрасывание признака по итогам процентного содержания пропусков
rejectionAttribute('Total Recovered')

# Внесение недостающих значений для всех числовых столбцов
addValues()
#makeHeatChart() #проверка заполненя

#-------------------------------------------------------------------
#проверка выбросов
#коробчатая гистограмма
makeBoxHistogram('Total Cases')
makeBoxHistogram('Total Deaths')
makeBoxHistogram('Total Recovered')
makeBoxHistogram('Active Cases')
makeBoxHistogram('Total Test')
makeBoxHistogram('Population')

# Описательная статистика
makeDescriptiveStatistics('Total Cases')
makeDescriptiveStatistics('Total Deaths')
makeDescriptiveStatistics('Total Recovered')
makeDescriptiveStatistics('Active Cases')
makeDescriptiveStatistics('Total Test')
makeDescriptiveStatistics('Population')


# Столбчатая диаграмма
makeBarChart('Total Cases')
makeBarChart('Total Deaths')
makeBarChart('Total Recovered')
makeBarChart('Active Cases')
makeBarChart('Total Test')
makeBarChart('Population')


#поиск часто повторяющихся значений признака
mostIsSame()

#отбрасываем повторяющиеся значения
findSameBySerialNumber('Serial Number')

#отбрасывание по критическим параметрам
findSameByCriticalParametersAndDelete()




