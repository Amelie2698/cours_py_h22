import sqlite3
from sqlite3 import connect

def __connect_db():
    conn = connect('base_donnees.db', check_same_thread = False)
    return conn

def __creer_tableau_(tableau):
    conn = __connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tab1 (sex, address, Fedu, Medu, famrel, G1, G2, G3)''')
    conn.commit()
    conn.close()
    
def __definir_les_donnees_(tableau, data_sql):
    conn = __connect_db()
    data_sql.to_sql('tab1', conn, if_exists='replace', index = False)
    conn.commit()
    conn.close()

