print("Hello World")

import pandas
print (pandas.__version__)
url = 'https://data.seattle.gov/resource/frxe-s3us.json'
df = pandas.read_json(url)
print(df)
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')


# pandas.read_excel('banane2.csv', engine = 'openpyxl')


print(df.columns)
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
print(df['department']) #donne même chose que .departement
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
print(df.department)
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
df.to_csv('banane3.csv')
