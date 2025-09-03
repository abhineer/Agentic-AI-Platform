# database.py
import sqlite3
import logging

def init_database(db_path='inventory.db'):
    """Initialize SQLite database with schema and sample data"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY,
                product_name TEXT UNIQUE NOT NULL,
                current_stock INTEGER NOT NULL,
                average_demand REAL NOT NULL,
                lead_time INTEGER NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sample data
        sample_data = [
            ('toothpaste', 20, 5.0, 4),
            ('shampoo', 15, 3.0, 5),
            ('soap', 50, 8.0, 3),
            ('detergent', 25, 4.5, 6),
            ('toilet_paper', 100, 12.0, 7),
            ('toothbrush', 30, 2.0, 3),
            ('mouthwash', 10, 1.0, 4),
            ('floss', 20, 1.5, 5),
            ('razor', 15, 0.5, 3),
            ('shaving_cream', 25, 1.0, 4),
            ('shaving_gel', 10, 0.5, 3),
            ('shaving_foam', 15, 1.0, 4),
            ('milk', 10, 2.0, 3),
            ('bread', 20, 1.0, 4),
            ('eggs', 30, 0.5, 3),
            ('cheese', 15, 1.0, 4),
            ('butter', 10, 0.5, 3),
            ('yogurt', 15, 1.0, 4),
            ('juice', 20, 2.0, 3)
                             
        ]
        
        # Insert sample data
        cursor.executemany('''
            INSERT OR IGNORE INTO inventory 
            (product_name, current_stock, average_demand, lead_time)
            VALUES (?, ?, ?, ?)
        ''', sample_data)
        
        conn.commit()
        logging.info("Database initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_db_connection(db_path='inventory.db'):
    """Get a database connection"""
    return sqlite3.connect(db_path)