"""
UNIFIED TRADING SYSTEM v2.3 - Cosmetic Updates Only
- Strategy 100% UNCHANGED: 9/21 SMA Crossover + Volume > 1.5x
- News integration 100% INTACT
- Added: Unrealised P&L display
- Added: CSV/XLSX/TXT portfolio import
- Removed: Balloon animation
- Added: GitHub storage persistence
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
import sqlite3
import io
import time
from typing import Dict, List, Tuple
import plotly.express as px
import numpy as np
import requests
import feedparser
from textblob import TextBlob
import base64
import json
from pathlib import Path

# ==================== TOKEN STORAGE FUNCTIONS ====================
def get_token_file():
    """Get path to token file"""
    return Path.home() / '.portfolio_token.json'

def save_token(token: str, repo: str):
    """Save GitHub token to file"""
    try:
        data = {'token': token, 'repo': repo}
        with open(get_token_file(), 'w') as f:
            json.dump(data, f)
        return True
    except:
        return False

def load_token():
    """Load saved GitHub token"""
    try:
        token_file = get_token_file()
        if token_file.exists():
            with open(token_file, 'r') as f:
                data = json.load(f)
                return data.get('token', ''), data.get('repo', '')
        return '', ''
    except:
        return '', ''

def delete_token():
    """Delete saved token"""
    try:
        token_file = get_token_file()
        if token_file.exists():
            token_file.unlink()
        return True
    except:
        return False
# Page config
st.set_page_config(
    page_title="Stock Scanner & Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .portfolio-tile {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .profit {
        color: #28a745;
        font-weight: bold;
    }
    .loss {
        color: #dc3545;
        font-weight: bold;
    }
    .pnl-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== GITHUB STORAGE MANAGER ====================

class GitHubStorageManager:
    """Manages persistent storage on GitHub"""
    
    def __init__(self):
        self.api_base = "https://api.github.com"
        self.token = None
        self.owner = None
        self.repo = None
        self.repo_name = None
        self.configured = False
        self.load_config()
    
    def load_config(self):
        """Load GitHub config from session state"""
        if 'github_token' in st.session_state and 'github_repo' in st.session_state:
            self.token = st.session_state.github_token
            self.repo = st.session_state.github_repo
            self.owner = st.session_state.get('github_owner', '')
            self.configured = bool(self.token and self.repo and self.owner)
    
    def save_to_github(self, filename: str, content: str, commit_message: str) -> bool:
        """Save file to GitHub repository"""
        if not self.configured:
            return False
        
        try:
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/contents/{filename}"
            headers = {
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers)
            sha = response.json().get('sha') if response.status_code == 200 else None
            
            content_base64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            data = {
                'message': commit_message,
                'content': content_base64,
                'branch': 'main'
            }
            
            if sha:
                data['sha'] = sha
            
            response = requests.put(url, headers=headers, json=data)
            return response.status_code in [200, 201]
            
        except Exception as e:
            st.error(f"GitHub save error: {str(e)}")
            return False
    
    def load_from_github(self, filename: str) -> str:
        """Load file from GitHub repository"""
        if not self.configured:
            return None
        
        try:
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/contents/{filename}"
            headers = {
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                content_base64 = response.json()['content']
                content_bytes = base64.b64decode(content_base64)
                return content_bytes.decode('utf-8')
            else:
                return None
                
        except Exception as e:
            return None


# ==================== NEWS AGGREGATOR WITH NEWSAPI ====================

class IndianNewsAggregator:
    """Multi-source news aggregator including NewsAPI"""
    
    def __init__(self, news_api_key="b4ced491f32745efa909fc97178bb9b1"):
        self.news_api_key = news_api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def fetch_google_news(self, symbol: str, company_name: str, max_articles: int = 3) -> List[Dict]:
        """Fetch news from Google News RSS"""
        articles = []
        try:
            queries = [company_name, symbol]
            
            for query in queries[:1]:
                url = f"https://news.google.com/rss/search?q={query}+india+stock&hl=en-IN&gl=IN&ceid=IN:en"
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:max_articles]:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    source = entry.get('source', {}).get('title', 'Google News')
                    
                    try:
                        pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                        date_str = pub_date.strftime('%Y-%m-%d')
                    except:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    text_lower = title.lower()
                    if symbol.lower() in text_lower or any(word in text_lower for word in company_name.lower().split()[:2]):
                        articles.append({
                            'title': title[:100],
                            'source': f"📰 {source}",
                            'date': date_str,
                            'url': link,
                            'provider': 'Google News'
                        })
                
                if articles:
                    break
        except:
            pass
        
        return articles[:max_articles]
    
    def fetch_economic_times(self, symbol: str, company_name: str, max_articles: int = 2) -> List[Dict]:
        """Fetch from Economic Times RSS"""
        articles = []
        try:
            et_feeds = [
                "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            ]
            
            for feed_url in et_feeds:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:15]:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    text_lower = title.lower()
                    if symbol.lower() in text_lower or any(word in text_lower for word in company_name.lower().split()[:2]):
                        try:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                            date_str = pub_date.strftime('%Y-%m-%d')
                        except:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        articles.append({
                            'title': title[:100],
                            'source': '📊 Economic Times',
                            'date': date_str,
                            'url': link,
                            'provider': 'Economic Times'
                        })
                
                if articles:
                    break
        except:
            pass
        
        return articles[:max_articles]
    
    def fetch_moneycontrol(self, symbol: str, company_name: str, max_articles: int = 2) -> List[Dict]:
        """Fetch from Moneycontrol RSS"""
        articles = []
        try:
            url = "https://www.moneycontrol.com/rss/latestnews.xml"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:20]:
                title = entry.get('title', '')
                link = entry.get('link', '')
                published = entry.get('published', '')
                
                text_lower = title.lower()
                if symbol.lower() in text_lower or any(word in text_lower for word in company_name.lower().split()[:2]):
                    try:
                        pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                        date_str = pub_date.strftime('%Y-%m-%d')
                    except:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    articles.append({
                        'title': title[:100],
                        'source': '💰 Moneycontrol',
                        'date': date_str,
                        'url': link,
                        'provider': 'Moneycontrol'
                    })
        except:
            pass
        
        return articles[:max_articles]
    
    def fetch_newsapi(self, symbol: str, company_name: str) -> List[Dict]:
        """Fetch from NewsAPI"""
        articles = []
        try:
            query = company_name if company_name != symbol else symbol
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.news_api_key,
                'q': f"{query} stock",
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': from_date,
                'pageSize': 5
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', [])[:3]:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    source_name = article.get('source', {}).get('name', 'Unknown')
                    url = article.get('url', '')
                    published = article.get('publishedAt', '')
                    
                    try:
                        pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                        date_str = pub_date.strftime('%Y-%m-%d')
                    except:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    text_combined = f"{title} {description}".lower()
                    if symbol.lower() in text_combined or company_name.lower() in text_combined:
                        articles.append({
                            'title': title[:100],
                            'source': f"📡 {source_name}",
                            'date': date_str,
                            'url': url,
                            'provider': 'NewsAPI'
                        })
        except:
            pass
        
        return articles
    
    def get_all_news(self, symbol: str, company_name: str, max_total: int = 8) -> List[Dict]:
        """Aggregate news from all sources"""
        all_articles = []
        
        all_articles.extend(self.fetch_newsapi(symbol, company_name))
        all_articles.extend(self.fetch_google_news(symbol, company_name))
        all_articles.extend(self.fetch_economic_times(symbol, company_name))
        all_articles.extend(self.fetch_moneycontrol(symbol, company_name))
        
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title_lower = article['title'].lower()[:50]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        unique_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return unique_articles[:max_total]


# ==================== DATABASE MANAGER ====================

class PortfolioDatabase:
    def __init__(self, db_path="trading_system.db"):
        self.db_path = db_path
        self.github = GitHubStorageManager()
        self.init_database()
        
        if self.github.configured:
            self.sync_from_github()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                invested_amount REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total_amount REAL NOT NULL,
                transaction_date DATE NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realized_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                quantity REAL NOT NULL,
                buy_price REAL NOT NULL,
                sell_price REAL NOT NULL,
                profit_loss REAL NOT NULL,
                profit_loss_pct REAL NOT NULL,
                transaction_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Add transaction_id column to existing databases
        cursor.execute("PRAGMA table_info(realized_pnl)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'transaction_id' not in columns:
            cursor.execute('ALTER TABLE realized_pnl ADD COLUMN transaction_id INTEGER')
        
        conn.commit()
        conn.close()
    
    def sync_to_github(self):
        """Sync portfolio data to GitHub"""
        if not self.github.configured:
            return False
        
        try:
            holdings = self.get_holdings()
            transactions = self.get_transactions(limit=1000)
            realized = self.get_realized_pnl()
            
            portfolio_data = {
                'holdings': holdings.to_dict('records') if not holdings.empty else [],
                'transactions': transactions.to_dict('records') if not transactions.empty else [],
                'realized_pnl': realized.to_dict('records') if not realized.empty else [],
                'last_sync': datetime.now().isoformat()
            }
            
            content = json.dumps(portfolio_data, indent=2, default=str)
            
            success = self.github.save_to_github(
                'portfolio_data.json',
                content,
                f'Auto-sync portfolio - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            
            return success
            
        except Exception as e:
            return False
    
    def sync_from_github(self):
        """Load portfolio data from GitHub"""
        if not self.github.configured:
            return False
        
        try:
            content = self.github.load_from_github('portfolio_data.json')
            
            if content:
                portfolio_data = json.loads(content)
                
                conn = sqlite3.connect(self.db_path)
                conn.execute("DELETE FROM holdings")
                conn.execute("DELETE FROM transactions")
                conn.execute("DELETE FROM realized_pnl")
                conn.commit()
                
                if portfolio_data.get('holdings'):
                    holdings_df = pd.DataFrame(portfolio_data['holdings'])
                    holdings_df.to_sql('holdings', conn, if_exists='append', index=False)
                
                if portfolio_data.get('transactions'):
                    trans_df = pd.DataFrame(portfolio_data['transactions'])
                    trans_df.to_sql('transactions', conn, if_exists='append', index=False)
                
                if portfolio_data.get('realized_pnl'):
                    realized_df = pd.DataFrame(portfolio_data['realized_pnl'])
                    realized_df.to_sql('realized_pnl', conn, if_exists='append', index=False)
                
                conn.close()
                return True
                
        except Exception as e:
            return False
    
    def bulk_import_transactions(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
    """Bulk import transactions from dataframe"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    success_count = 0
    errors = []
    
    # Sort by date to process in chronological order
    df = df.sort_values('Date')
    
    for idx, row in df.iterrows():
        try:
            symbol = str(row.get('Symbol', '')).upper().strip()
            company = str(row.get('Company', symbol))
            trans_type = str(row.get('Type', 'BUY')).upper().strip()
            quantity = float(row.get('Quantity', 0))
            price = float(row.get('Price', 0))
            trans_date = row.get('Date')
            notes = str(row.get('Notes', ''))
            
            # Validate
            if not symbol or quantity <= 0 or price <= 0:
                errors.append(f"Row {idx+1}: Invalid data")
                continue
            
            if trans_type not in ['BUY', 'SELL']:
                errors.append(f"Row {idx+1}: Type must be BUY or SELL, got '{trans_type}'")
                continue
            
            # Add transaction
            total_amount = quantity * price
            
            cursor.execute('''
                INSERT INTO transactions (symbol, company_name, transaction_type,
                                         quantity, price, total_amount, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, company, trans_type, quantity, price, total_amount, trans_date, notes))
            
            transaction_id = cursor.lastrowid
            
            # Update holdings
            if trans_type == 'BUY':
                realized_pnl = self._add_to_holdings(cursor, symbol, company, quantity, price)
                if realized_pnl:
                    cursor.execute('''
                        INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price,
                                                 sell_price, profit_loss, profit_loss_pct, transaction_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', realized_pnl + (transaction_id,))
            elif trans_type == 'SELL':
                realized_pnl = self._reduce_from_holdings(cursor, symbol, quantity, price)
                if realized_pnl:
                    cursor.execute('''
                        INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price,
                                                 sell_price, profit_loss, profit_loss_pct, transaction_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', realized_pnl + (transaction_id,))
            
            success_count += 1
            
        except Exception as e:
            errors.append(f"Row {idx+1}: {str(e)}")
    
    conn.commit()
    conn.close()
    
    if self.github.configured:
        self.sync_to_github()
    
    return success_count, errors
    
    def add_transaction(self, symbol: str, company_name: str, trans_type: str,
                       quantity: float, price: float, trans_date: date, notes: str = ''):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_amount = quantity * price
        
        cursor.execute('''
            INSERT INTO transactions (symbol, company_name, transaction_type,
                                     quantity, price, total_amount, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol.upper(), company_name, trans_type.upper(),
              quantity, price, total_amount, trans_date, notes))
        
        transaction_id = cursor.lastrowid
        
        if trans_type.upper() == 'BUY':
            realized_pnl = self._add_to_holdings(cursor, symbol, company_name, quantity, price)
            # Record realized P&L if buying to cover a short position
            if realized_pnl:
                cursor.execute('''
                    INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price,
                                             sell_price, profit_loss, profit_loss_pct, transaction_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', realized_pnl + (transaction_id,))
        elif trans_type.upper() == 'SELL':
            realized_pnl = self._reduce_from_holdings(cursor, symbol, quantity, price)
            # Record realized P&L if closing a long position
            if realized_pnl:
                cursor.execute('''
                    INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price,
                                             sell_price, profit_loss, profit_loss_pct, transaction_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', realized_pnl + (transaction_id,))
        conn.commit()
        conn.close()
        
        if self.github.configured:
            self.sync_to_github()
        
        return transaction_id
                           
    def _add_to_holdings(self, cursor, symbol: str, company_name: str,
                        quantity: float, price: float):
        cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
        holding = cursor.fetchone()
        
        if holding:
            old_qty = holding[3]
            old_avg = holding[4]
            
            # Check if covering a SHORT position
            if old_qty < 0:
                # Covering short position - calculate P&L
                cover_qty = min(quantity, abs(old_qty))  # How many shares we're covering
                profit_loss = (old_avg - price) * cover_qty  # Profit when price drops
                profit_loss_pct = ((old_avg - price) / old_avg) * 100 if old_avg != 0 else 0
                
                new_qty = old_qty + quantity
                
                if new_qty == 0:
                    # Fully covered short
                    cursor.execute('DELETE FROM holdings WHERE symbol = ?', (symbol.upper(),))
                elif new_qty < 0:
                    # Still short, but reduced
                    new_invested = new_qty * old_avg
                    cursor.execute('''
                        UPDATE holdings
                        SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (new_qty, new_invested, symbol.upper()))
                else:
                    # Covered short and now LONG
                    remaining_qty = new_qty
                    new_avg = price
                    new_invested = remaining_qty * new_avg
                    cursor.execute('''
                        UPDATE holdings
                        SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (new_qty, new_avg, new_invested, symbol.upper()))
                
                # Return realized P&L (sell_price is the short open price, buy_price is cover price)
                return (symbol.upper(), holding[2], cover_qty, price, old_avg, profit_loss, profit_loss_pct)
            
            else:
                # Adding to LONG position - normal calculation
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
                new_invested = new_qty * new_avg
                
                cursor.execute('''
                    UPDATE holdings
                    SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (new_qty, new_avg, new_invested, symbol.upper()))
                
                return None  # No realized P&L
        else:
            # New LONG position
            invested = quantity * price
            cursor.execute('''
                INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol.upper(), company_name, quantity, price, invested))
            
            return None  # No realized P&L
    
    def _reduce_from_holdings(self, cursor, symbol: str, quantity: float,
                                 sell_price: float) -> Tuple:
            cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
            holding = cursor.fetchone()
            
            # If no holding exists, create SHORT position (negative quantity)
            if not holding:
                st.info(f"📉 Opening SHORT position for {symbol.upper()}")
                # Create new holding with NEGATIVE quantity
                new_qty = -quantity
                #this is where I have changed to test
                invested = new_qty * sell_price
                cursor.execute('''
                    INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                    VALUES (?, ?, ?, ?, ?)
                ''', (symbol.upper(), symbol, new_qty, sell_price, invested))
                
                # Return None - no realized P&L when opening short
                return None
            
            current_qty = holding[3]
            avg_price = holding[4]
            company_name = holding[2]
            
            # Calculate P&L based on position type
            if current_qty > 0:
                # LONG position - profit when price goes UP
                profit_loss = (sell_price - avg_price) * quantity
                profit_loss_pct = ((sell_price - avg_price) / avg_price) * 100
            else:
                # SHORT position - profit when price goes DOWN
                profit_loss = (avg_price - sell_price) * quantity
                profit_loss_pct = ((avg_price - sell_price) / avg_price) * 100
            
            new_qty = current_qty - quantity
            
            # Update or delete holding
            if new_qty == 0:
                # Position fully closed
                cursor.execute('DELETE FROM holdings WHERE symbol = ?', (symbol.upper(),))
            else:
                # Position still open (may be negative for short)
                new_invested = new_qty * avg_price
                cursor.execute('''
                    UPDATE holdings
                    SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (new_qty, new_invested, symbol.upper()))
            
            return (symbol.upper(), company_name, quantity, avg_price, sell_price,
                    profit_loss, profit_loss_pct)
    
    def delete_holding(self, holding_id: int):
        """Delete a holding entirely"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM holdings WHERE id = ?', (holding_id,))
        conn.commit()
        conn.close()
        
        if self.github.configured:
            self.sync_to_github()
    
    def update_holding(self, holding_id: int, quantity: float, avg_price: float):
        """Update holding quantity and average price"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        invested = quantity * avg_price
        cursor.execute('''
            UPDATE holdings
            SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (quantity, avg_price, invested, holding_id))
        conn.commit()
        conn.close()
        
        if self.github.configured:
            self.sync_to_github()
    
    def get_holdings(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM holdings ORDER BY updated_at DESC', conn)
        conn.close()
        return df
    
    def get_transactions(self, limit: int = 50) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM transactions ORDER BY transaction_date DESC, created_at DESC LIMIT {limit}',
            conn
        )
        conn.close()
        return df
    
    def get_realized_pnl(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM realized_pnl ORDER BY created_at DESC', conn)
        conn.close()
        return df
    
    def get_portfolio_summary(self) -> Dict:
        holdings = self.get_holdings()
        
        return {
            'total_holdings': len(holdings),
            'total_invested': holdings['invested_amount'].sum() if not holdings.empty else 0,
            'total_quantity': holdings['quantity'].sum() if not holdings.empty else 0
        }
    
    def bulk_import_portfolio(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """Bulk import portfolio from dataframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        success_count = 0
        errors = []
        
        for idx, row in df.iterrows():
            try:
                symbol = str(row.get('Symbol', '')).upper().strip()
                company = str(row.get('Company', symbol))
                quantity = float(row.get('Quantity', 0))
                avg_price = float(row.get('Avg Price', 0))
                
                if not symbol or quantity <= 0 or avg_price <= 0:
                    errors.append(f"Row {idx+1}: Invalid data")
                    continue
                
                invested = quantity * avg_price
                
                cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    cursor.execute('''
                        UPDATE holdings
                        SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (quantity, avg_price, invested, symbol))
                else:
                    cursor.execute('''
                        INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, company, quantity, avg_price, invested))
                
                success_count += 1
                
            except Exception as e:
                errors.append(f"Row {idx+1}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        if self.github.configured:
            self.sync_to_github()
        
        return success_count, errors


# ==================== EXCEL/CSV/TXT PROCESSORS ====================

def load_stock_list_from_file(file) -> List[str]:
    """Load stock symbols from Excel, CSV, or TXT file"""
    try:
        filename = file.name.lower()
        
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
            symbol_col = None
            for col in ['Symbol', 'symbol', 'SYMBOL', 'Stock', 'Ticker']:
                if col in df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                symbols = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
            else:
                symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
        
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
            symbol_col = None
            for col in ['Symbol', 'symbol', 'SYMBOL', 'Stock', 'Ticker']:
                if col in df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                symbols = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
            else:
                symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
        
        else:  # TXT file
            content = file.read().decode('utf-8')
            symbols = [line.strip().upper() for line in content.split('\n') if line.strip()]
        
        symbols = [s.replace('.NS', '').replace('.BO', '') for s in symbols if s]
        return list(set(symbols))
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return []


def load_portfolio_from_file(file) -> pd.DataFrame:
    """Load portfolio from Excel, CSV, or TXT file"""
    try:
        filename = file.name.lower()
        
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            from io import StringIO
            df = pd.read_csv(StringIO(content))
        else:
            st.error("Unsupported file format")
            return pd.DataFrame()
        
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'symbol' in col_lower or 'stock' in col_lower or 'ticker' in col_lower:
                col_map[col] = 'Symbol'
            elif 'company' in col_lower or 'name' in col_lower:
                col_map[col] = 'Company'
            elif 'quantity' in col_lower or 'qty' in col_lower or 'shares' in col_lower:
                col_map[col] = 'Quantity'
            elif 'price' in col_lower or 'avg' in col_lower or 'cost' in col_lower:
                col_map[col] = 'Avg Price'
        
        df.rename(columns=col_map, inplace=True)
        
        if 'Symbol' not in df.columns or 'Quantity' not in df.columns or 'Avg Price' not in df.columns:
            st.error("Required columns: Symbol, Quantity, Avg Price")
            return pd.DataFrame()
        
        if 'Company' not in df.columns:
            df['Company'] = df['Symbol']
        
        return df[['Symbol', 'Company', 'Quantity', 'Avg Price']]
        
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return pd.DataFrame()

def load_transactions_from_file(file) -> pd.DataFrame:
    """Load transaction history from Excel, CSV, or TXT file"""
    try:
        filename = file.name.lower()
        
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            from io import StringIO
            df = pd.read_csv(StringIO(content))
        else:
            st.error("Unsupported file format")
            return pd.DataFrame()
        
        # Map column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'symbol' in col_lower or 'stock' in col_lower or 'ticker' in col_lower:
                col_map[col] = 'Symbol'
            elif 'company' in col_lower or 'name' in col_lower:
                col_map[col] = 'Company'
            elif 'quantity' in col_lower or 'qty' in col_lower or 'shares' in col_lower:
                col_map[col] = 'Quantity'
            elif 'price' in col_lower or 'rate' in col_lower:
                col_map[col] = 'Price'
            elif 'type' in col_lower or 'action' in col_lower or 'side' in col_lower:
                col_map[col] = 'Type'
            elif 'date' in col_lower:
                col_map[col] = 'Date'
            elif 'note' in col_lower or 'remark' in col_lower or 'comment' in col_lower:
                col_map[col] = 'Notes'
        
        df.rename(columns=col_map, inplace=True)
        
        # Validate required columns
        required = ['Symbol', 'Quantity', 'Price', 'Type', 'Date']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.info("Required: Symbol, Quantity, Price, Type (BUY/SELL), Date")
            return pd.DataFrame()
        
        # Add Company if missing
        if 'Company' not in df.columns:
            df['Company'] = df['Symbol']
        
        # Add Notes if missing
        if 'Notes' not in df.columns:
            df['Notes'] = ''
        
        # Clean up Type column
        df['Type'] = df['Type'].str.upper().str.strip()
        
        # Convert Date to proper format
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        return df[['Symbol', 'Company', 'Type', 'Quantity', 'Price', 'Date', 'Notes']]
        
    except Exception as e:
        st.error(f"Error loading transactions: {str(e)}")
        return pd.DataFrame()
# ==================== STOCK ANALYZER - STRATEGY 100% INTACT ====================

def analyze_stock(symbol: str, company_name: str) -> Dict:
    """
    COMPLETE WORKING VERSION - Detects both Bullish and Bearish signals
    Strategy: SMA 9/21 crossover within last 5 days + High Volume (1.5x+)
    """
    try:
        # Fetch stock data
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 30:
            return None
        
        # Calculate SMAs
        hist['SMA9'] = hist['Close'].rolling(window=9).mean()
        hist['SMA21'] = hist['Close'].rolling(window=21).mean()
        
        # Get latest values
        latest = hist.iloc[-1]
        current_price = latest['Close']
        sma9 = latest['SMA9']
        sma21 = latest['SMA21']
        
        if pd.isna(sma9) or pd.isna(sma21):
            return None
        
        # Calculate 21-day volume average (excluding current day)
        vol_21day_avg = hist['Volume'].iloc[-22:-1].mean() if len(hist) >= 22 else hist['Volume'].iloc[:-1].mean()
        
        # Current trend
        current_trend = 'BULLISH' if sma9 > sma21 else 'BEARISH'
        
        # ============= CROSSOVER DETECTION - WORKING VERSION =============
        bullish_crossover_detected = False
        bearish_crossover_detected = False
        bullish_crossover_day = None
        bearish_crossover_day = None
        
        # Check all 5 days (NO BREAK - must check all days!)
        for i in range(1, min(6, len(hist))):
            prev = hist.iloc[-(i+1)]
            curr = hist.iloc[-i]
            
            # Check both SMAs are valid
            if pd.notna(prev['SMA9']) and pd.notna(prev['SMA21']) and \
               pd.notna(curr['SMA9']) and pd.notna(curr['SMA21']):
                
                # Check for BULLISH crossover (SMA9 crosses above SMA21)
                if prev['SMA9'] <= prev['SMA21'] and curr['SMA9'] > curr['SMA21']:
                    if not bullish_crossover_detected:
                        bullish_crossover_detected = True
                        bullish_crossover_day = i
                
                # Check for BEARISH crossover (SMA9 crosses below SMA21)
                # Use 'if' not 'elif' - must check independently!
                if prev['SMA9'] >= prev['SMA21'] and curr['SMA9'] < curr['SMA21']:
                    if not bearish_crossover_detected:
                        bearish_crossover_detected = True
                        bearish_crossover_day = i
        
        # Validate crossovers are still active (CRITICAL STEP)
        if bullish_crossover_detected and sma9 <= sma21:
            # Bullish crossover happened but SMA9 has fallen back below SMA21
            bullish_crossover_detected = False
            bullish_crossover_day = None
        
        if bearish_crossover_detected and sma9 >= sma21:
            # Bearish crossover happened but SMA9 has risen back above SMA21
            bearish_crossover_detected = False
            bearish_crossover_day = None
        
        # Determine final crossover status
        crossover_detected = bullish_crossover_detected or bearish_crossover_detected
        
        if bullish_crossover_detected:
            crossover_type = 'BULLISH'
            crossover_day = bullish_crossover_day
        elif bearish_crossover_detected:
            crossover_type = 'BEARISH'
            crossover_day = bearish_crossover_day
        else:
            crossover_type = None
            crossover_day = None
        # ============= END CROSSOVER DETECTION =============
        
        # ============= VOLUME ANALYSIS - ORIGINAL LOGIC =============
        today_volume = hist['Volume'].iloc[-1]
        yesterday_volume = hist['Volume'].iloc[-2] if len(hist) >= 2 else 0
        
        high_volume_today = today_volume > vol_21day_avg * 1.5
        high_volume_yesterday = yesterday_volume > vol_21day_avg * 1.5
        high_volume = high_volume_today or high_volume_yesterday
        
        volume_ratio_today = today_volume / vol_21day_avg if vol_21day_avg > 0 else 0
        volume_ratio_yesterday = yesterday_volume / vol_21day_avg if vol_21day_avg > 0 else 0
        # ============= END VOLUME ANALYSIS =============
        
        # Return complete result
        return {
            'symbol': symbol,
            'company_name': company_name,
            'current_price': current_price,
            'sma9': sma9,
            'sma21': sma21,
            'trend': current_trend,
            'crossover_detected': crossover_detected,
            'crossover_type': crossover_type,
            'crossover_day': crossover_day,
            'volume_ratio': volume_ratio_today,
            'volume_ratio_yesterday': volume_ratio_yesterday,
            'high_volume': high_volume,
            'high_volume_today': high_volume_today,
            'high_volume_yesterday': high_volume_yesterday,
            'today_volume': today_volume,
            'yesterday_volume': yesterday_volume
        }
        
    except Exception as e:
        return None


def get_stock_info(symbol: str) -> Dict:
    """Get basic stock info"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period='1d')
        return {
            'name': info.get('longName', symbol),
            'current_price': hist['Close'].iloc[-1] if not hist.empty else 0,
            'valid': True
        }
    except:
        return {'name': symbol, 'current_price': 0, 'valid': False}


def format_currency(amount: float) -> str:
    """Format currency"""
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:,.2f}"


def format_volume(volume: float) -> str:
    """Format volume for display"""
    if volume >= 1e7:
        return f"{volume/1e7:.2f}Cr"
    elif volume >= 1e5:
        return f"{volume/1e5:.2f}L"
    elif volume >= 1e3:
        return f"{volume/1e3:.2f}K"
    else:
        return f"{volume:.0f}"

def detect_instrument_type(symbol: str) -> tuple:
    """
    Detect instrument type and return (type, yahoo_symbol)
    Types: 'equity', 'nifty_future', 'commodity', 'index'
    """
    symbol_upper = symbol.upper()
    
    # GIFT Nifty / Nifty Futures
    if 'NIFTY' in symbol_upper and ('FUT' in symbol_upper or 'GIFT' in symbol_upper):
        return ('nifty_future', 'NIFTY50.NS')  # Use Nifty spot as proxy
    
    # Nifty Index (spot)
    if symbol_upper == 'NIFTY' or symbol_upper == 'NIFTY50':
        return ('index', '^NSEI')
    
    # Bank Nifty
    if symbol_upper == 'BANKNIFTY' or symbol_upper == 'BANKNIFTY50':
        return ('index', '^NSEBANK')
    
    # MCX Commodities
    if any(commodity in symbol_upper for commodity in ['SILVER', 'GOLD', 'CRUDE', 'COPPER', 'ZINC']):
        return ('commodity', symbol_upper)
    
    # Regular equity
    return ('equity', f"{symbol_upper}.NS")

def get_current_price(symbol: str) -> float:
    """
    Get current price for any instrument type
    """
    instrument_type, yahoo_symbol = detect_instrument_type(symbol)
    
    try:
        if instrument_type in ['equity', 'index', 'nifty_future']:
            # Use Yahoo Finance
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]
        
        elif instrument_type == 'commodity':
            # For commodities, return a placeholder
            # You'll need to integrate with MCX API or manual entry
            st.warning(f"⚠️ {symbol}: Commodity prices not auto-fetched. Using manual price.")
            return 0.0
        
        return 0.0
    
    except Exception as e:
        st.error(f"Error fetching price for {symbol}: {str(e)}")
        return 0.0


# ==================== MAIN APP ====================

def main():
    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = PortfolioDatabase()
    
    db = st.session_state.db
    
    # Auto-load saved GitHub token (NEW CODE)
    if 'token_loaded' not in st.session_state:
        saved_token, saved_repo = load_token()
        if saved_token and saved_repo:
            # Parse repo into owner and name
            if '/' in saved_repo:
                owner, repo = saved_repo.split('/', 1)
                db.github.owner = owner
                db.github.repo = repo
                db.github.repo_name = saved_repo
                db.github.token = saved_token
                db.github.configured = True
        st.session_state.token_loaded = True
        
    # Initialize news aggregator with NewsAPI
    if 'news_aggregator' not in st.session_state:
        st.session_state.news_aggregator = IndianNewsAggregator()
    
    # Initialize stock list
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = []
    
    if 'temp_symbols' not in st.session_state:
        st.session_state.temp_symbols = []
    
    news_agg = st.session_state.news_aggregator
    
    # Load stock list from GitHub if available
    if db.github.configured and not st.session_state.stock_list:
        content = db.github.load_from_github('stock_symbols.txt')
        if content:
            st.session_state.stock_list = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Sidebar
    st.sidebar.title("📊 Stock Scanner & Portfolio Manager")
    
    # GitHub Configuration
    #with st.sidebar.expander("☁️ GitHub Storage", expanded=False):
    #    github_owner = st.text_input("GitHub Username")
    #    github_repo = st.text_input("Repository Name")
    #    github_token = st.text_input("Access Token", type="password")
        
    # GitHub Configuration
    with st.sidebar.expander("☁️ GitHub Storage", expanded=False):
        
        if db.github.configured:
            # Connected - show status and actions
            st.success("✅ Connected")
            st.caption(f"📁 Repository: {db.github.repo_name}")
            
            # Sync buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬆️ Sync"):
                    db.sync_to_github()
                    st.success("Synced!")
            with col2:
                if st.button("⬇️ Load"):
                    db.sync_from_github()
                    st.rerun()
            
            # Disconnect
            st.divider()
            if st.button("🔓 Disconnect"):
                delete_token()
                db.github.token = None
                db.github.repo_name = None
                db.github.configured = False
                if 'token_loaded' in st.session_state:
                    del st.session_state.token_loaded
                st.rerun()
        
        else:
            # Not connected - show form
            st.warning("⚠️ Not Connected")
            
            github_repo = st.text_input(
                "Repository Name",
                placeholder="username/repo-name",
                help="Format: your-username/your-repository"
            )
            
            github_token = st.text_input(
                "Access Token",
                type="password",
                help="Generate at: github.com/settings/tokens"
            )
                        
            if st.button("💾 Connect & Save", type="primary"):
                if github_token and github_repo:
                    # Save token to file
                    if save_token(github_token, github_repo):
                        # Parse repo into owner and name
                        if '/' in github_repo:
                            owner, repo = github_repo.split('/', 1)
                            db.github.owner = owner
                            db.github.repo = repo
                            db.github.repo_name = github_repo
                        else:
                            st.error("⚠️ Repo format must be: username/repo-name")
                            st.stop()
                        
                        db.github.token = github_token
                        db.github.configured = True
            else:
                st.error("⚠️ Please enter both token and repository name")
        
        st.caption("💡 Token is saved locally and persists forever")
        st.sidebar.divider()
    
    with st.sidebar.expander("🗑️ Database Management", expanded=False):
        st.warning("⚠️ **Reset Database**")
        st.caption("Use this to fix database schema errors. This will delete the local database file and recreate it.")
        
        if st.button("🔄 Reset Database Schema", type="primary"):
            import os
            try:
                # Close any connections
                if 'db' in st.session_state:
                    del st.session_state.db
                
                # Delete database file
                if os.path.exists("trading_system.db"):
                    os.remove("trading_system.db")
                    st.success("✅ Database reset! Reloading app...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("No database file found")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📈 Stock Scanner", ...]
    )            
    page = st.sidebar.radio(
        "Navigation",
        ["📈 Stock Scanner", "💼 Portfolio Manager", "➕ Add Transaction",
         "📜 Transaction History", "💰 Realized P&L", "📤 Upload Files"]
    )
    
    st.sidebar.divider()
    
    st.markdown("<h1 class='main-header'>🚀 Unified Stock Scanner </h1>", unsafe_allow_html=True)
    
    # ==================== STOCK SCANNER ====================
    if page == "📈 Stock Scanner":
        st.header("Stock Scanner - SMA + Volume")
        
        st.info("📋 Upload stock list (Excel/CSV/TXT) or load from GitHub")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Stock List",
                type=['xlsx', 'xls', 'csv', 'txt'],
                help="Upload Excel, CSV, or TXT file with stock symbols"
            )
        
        with col2:
            if st.button("📥 Use GitHub List", disabled=not db.github.configured):
                content = db.github.load_from_github('stock_symbols.txt')
                if content:
                    st.session_state.stock_list = [line.strip() for line in content.split('\n') if line.strip()]
                    st.success(f"✅ Loaded {len(st.session_state.stock_list)} symbols")
        
        if uploaded_file:
            symbols = load_stock_list_from_file(uploaded_file)
            if symbols:
                st.session_state.stock_list = symbols
                
                # Save to GitHub
                if db.github.configured:
                    content = '\n'.join(symbols)
                    db.github.save_to_github('stock_symbols.txt', content, 'Updated stock list')
                    st.success(f"✅ Loaded {len(symbols)} symbols (saved to GitHub)")
                else:
                    st.success(f"✅ Loaded {len(symbols)} symbols")
        
        if st.session_state.stock_list:
            st.info(f"📊 {len(st.session_state.stock_list)} stocks loaded")
            
            if st.button("🔍 Start Scanning", type="primary"):
                # Separate lists for bullish and bearish
                bullish_signals = []
                bearish_signals = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, symbol in enumerate(st.session_state.stock_list):
                    progress = (idx + 1) / len(st.session_state.stock_list)
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning {symbol}... ({idx+1}/{len(st.session_state.stock_list)})")
                    
                    stock_info = get_stock_info(symbol)
                    if stock_info['valid']:
                        analysis = analyze_stock(symbol, stock_info['name'])
                        if analysis:
                            # Collect both bullish and bearish signals
                            if analysis['crossover_detected'] and analysis['high_volume']:
                                if analysis['crossover_type'] == 'BULLISH':
                                    bullish_signals.append(analysis)
                                elif analysis['crossover_type'] == 'BEARISH':
                                    bearish_signals.append(analysis)
                    
                    time.sleep(0.3)
                
                progress_bar.empty()
                status_text.empty()
                
                if bullish_signals or bearish_signals:
                    st.success(f"✅ Found {len(bullish_signals)} bullish and {len(bearish_signals)} bearish signals!")
                    
                    # Display Bullish Signals Table
                    if bullish_signals:
                        st.subheader("🟢 Bullish Signals")
                        df_bullish = pd.DataFrame(bullish_signals)
                        st.dataframe(
                            df_bullish[['symbol', 'company_name', 'current_price', 'sma9', 'sma21',
                                      'crossover_day', 'volume_ratio', 'trend']],
                            use_container_width=True
                        )
                    
                    # Display Bearish Signals Table
                    if bearish_signals:
                        st.subheader("🔴 Bearish Signals")
                        df_bearish = pd.DataFrame(bearish_signals)
                        st.dataframe(
                            df_bearish[['symbol', 'company_name', 'current_price', 'sma9', 'sma21',
                                      'crossover_day', 'volume_ratio', 'trend']],
                            use_container_width=True
                        )
                    
                    # Show detailed results with news
                    st.divider()
                    
                    # ==================== BULLISH SIGNALS - DETAILED ====================
                    if bullish_signals:
                        st.subheader("🟢 Detailed Bullish Analysis with News & Recommendations")
                        
                        for result in bullish_signals:
                            with st.expander(f"🟢 {result['symbol']} - {result['company_name']}", expanded=True):
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Price", f"₹{result['current_price']:.2f}")
                                    st.metric("SMA 9", f"₹{result['sma9']:.2f}")
                                
                                with col2:
                                    st.metric("SMA 21", f"₹{result['sma21']:.2f}")
                                    st.metric("Trend", result['trend'])
                                
                                with col3:
                                    volume_emoji = "🔥" if result['high_volume_today'] else ""
                                    st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}x {volume_emoji}")
                                    st.metric("Crossover", f"{result['crossover_day']} days ago")
                                
                                st.divider()
                                
                                # Volume Details
                                st.subheader("📊 Volume Analysis")
                                
                                st.write(f"**Today:** {format_volume(result['today_volume'])} ({result['volume_ratio']:.2f}x of 21-day avg)")
                                if result['high_volume_today']:
                                    st.success("🔥 HIGH VOLUME TODAY - Strong buying interest!")
                                
                                st.write(f"**Yesterday:** {format_volume(result['yesterday_volume'])} ({result['volume_ratio_yesterday']:.2f}x of 21-day avg)")
                                if result['high_volume_yesterday']:
                                    st.success("🔥 HIGH VOLUME YESTERDAY")
                                
                                st.divider()
                                
                                # NEWS SECTION - SIMPLIFIED (NO SENTIMENT)
                                st.subheader("📰 Latest News")
                                
                                news_articles = news_agg.get_all_news(result['symbol'], result['company_name'])
                                
                                if news_articles:
                                    for article in news_articles:
                                        st.markdown(f"**{article['source']}** | {article['date']}")
                                        st.markdown(f"[{article['title']}]({article['url']})")
                                        st.markdown("---")
                                else:
                                    st.info("No recent news found")
                                
                                st.divider()
                                
                                # TRADING RECOMMENDATION - BULLISH
                                st.subheader("💡 Trading Recommendation")
                                st.success("🎯 **Action:** Strong BUY opportunity with technical confirmation")
                                st.write(f"📍 **Entry:** Around current levels (₹{result['current_price']:.2f}) or on minor pullback")
                                st.write(f"🛑 **Stop Loss:** Below ₹{result['sma21']:.2f} (21 SMA) or recent swing low")
                                st.write(f"🎁 **Risk/Reward:** Favorable with volume confirmation")
                                st.info("⚠️ **Note:** Always use proper position sizing and risk management")
                    
                    # ==================== BEARISH SIGNALS - DETAILED ====================
                    if bearish_signals:
                        st.subheader("🔴 Detailed Bearish Analysis with News & Recommendations")
                        
                        for result in bearish_signals:
                            with st.expander(f"🔴 {result['symbol']} - {result['company_name']}", expanded=True):
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Price", f"₹{result['current_price']:.2f}")
                                    st.metric("SMA 9", f"₹{result['sma9']:.2f}")
                                
                                with col2:
                                    st.metric("SMA 21", f"₹{result['sma21']:.2f}")
                                    st.metric("Trend", result['trend'])
                                
                                with col3:
                                    volume_emoji = "🔥" if result['high_volume_today'] else ""
                                    st.metric("Volume Ratio", f"{result['volume_ratio']:.2f}x {volume_emoji}")
                                    st.metric("Crossover", f"{result['crossover_day']} days ago")
                                
                                st.divider()
                                
                                # Volume Details
                                st.subheader("📊 Volume Analysis")
                                
                                st.write(f"**Today:** {format_volume(result['today_volume'])} ({result['volume_ratio']:.2f}x of 21-day avg)")
                                if result['high_volume_today']:
                                    st.error("🔥 HIGH VOLUME TODAY - Strong selling pressure!")
                                
                                st.write(f"**Yesterday:** {format_volume(result['yesterday_volume'])} ({result['volume_ratio_yesterday']:.2f}x of 21-day avg)")
                                if result['high_volume_yesterday']:
                                    st.error("🔥 HIGH VOLUME YESTERDAY")
                                
                                st.divider()
                                
                                # NEWS SECTION - SIMPLIFIED (NO SENTIMENT)
                                st.subheader("📰 Latest News")
                                
                                news_articles = news_agg.get_all_news(result['symbol'], result['company_name'])
                                
                                if news_articles:
                                    for article in news_articles:
                                        st.markdown(f"**{article['source']}** | {article['date']}")
                                        st.markdown(f"[{article['title']}]({article['url']})")
                                        st.markdown("---")
                                else:
                                    st.info("No recent news found")
                                
                                st.divider()
                                
                                # TRADING RECOMMENDATION - BEARISH
                                st.subheader("💡 Trading Recommendation")
                                st.error("🎯 **Action:** Consider SELL/SHORT with proper risk management")
                                st.write(f"📍 **Entry:** Around current levels (₹{result['current_price']:.2f}) or on minor bounce")
                                st.write(f"🛑 **Stop Loss:** Above ₹{result['sma21']:.2f} (21 SMA) or recent swing high")
                                st.write(f"⚠️ **Caution:** Monitor volume for continuation confirmation")
                                st.info("⚠️ **Note:** Always use proper position sizing and risk management")
                
                else:
                    st.warning("⚠️ No stocks found with crossover signals matching criteria")
    
    # ==================== PORTFOLIO MANAGER ====================
    elif page == "💼 Portfolio Manager":
        st.header("Portfolio Manager")
        
        # Auto-refresh toggle
        col1, col2 = st.columns([1, 3])
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
        with col2:
            if auto_refresh:
                refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
        
        # Refresh button
        col_refresh, col_clear = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
            # ==================== UNREALISED P&L SUMMARY ====================
            st.markdown("### 📊 Portfolio Summary")
            
            total_invested = 0
            total_current_value = 0
            portfolio_data = []
            
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    #current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_price = get_current_price(row['symbol'])
                    if current_price == 0:
                        continue  # Skip if price fetch failed
                        
                    # Handle both LONG and SHORT positions
                    if row['quantity'] > 0:
                        # LONG position - normal calculation
                        current_value = current_price * row['quantity']
                        unrealised_pnl = current_value - row['invested_amount']
                    else:
                        # SHORT position - reversed calculation
                        # Profit when price drops, loss when price rises
                        current_value = current_price * row['quantity'] * -1 # Will be negative
                        unrealised_pnl = - current_value - row['invested_amount'] # Reversed
                    
                    unrealised_pnl_pct = (unrealised_pnl / abs(row['invested_amount'])) * 100
                    
                    total_invested += abs(row['invested_amount'])
                    total_current_value += current_value
                    
                    portfolio_data.append({
                        'symbol': row['symbol'],
                        'invested': row['invested_amount'],
                        'current': current_value,
                        'pnl': unrealised_pnl,
                        'pnl_pct': unrealised_pnl_pct
                    })
                except:
                    pass
            
            if portfolio_data:
                #total_unrealised_pnl = total_current_value - total_invested
                total_unrealised_pnl = sum([p['pnl'] for p in portfolio_data])
                total_unrealised_pnl_pct = (total_unrealised_pnl / total_invested * 100) if total_invested > 0 else 0
                
                st.markdown("""
                <div class='pnl-summary'>
                    <h2 style='text-align: center; margin-bottom: 20px;'>💰 Unrealised Profit/Loss</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Invested", format_currency(total_invested))
                with col2:
                    st.metric("Current Value", format_currency(total_current_value))
                with col3:
                    pnl_color = "🟢" if total_unrealised_pnl >= 0 else "🔴"
                    st.metric(f"{pnl_color} Unrealised P&L", 
                             format_currency(total_unrealised_pnl),
                             f"{total_unrealised_pnl_pct:+.2f}%")
                with col4:
                    num_holdings = len(holdings)
                    winners = len([p for p in portfolio_data if p['pnl'] > 0])
                    st.metric("Holdings", num_holdings, f"{winners} profitable")
                
                st.divider()
            
            # Sidebar for Edit/Delete
            with st.sidebar:
                st.divider()
                st.subheader("✏️ Edit Holding")
                
                edit_options = ["-- Select --"] + [f"{row['symbol']} ({row['id']})" for _, row in holdings.iterrows()]
                edit_selection = st.selectbox("Choose holding to edit", edit_options, key="edit_select")
                
                if edit_selection != "-- Select --":
                    edit_id = int(edit_selection.split('(')[1].strip(')'))
                    edit_row = holdings[holdings['id'] == edit_id].iloc[0]
                    
                    st.write(f"**Editing: {edit_row['symbol']}**")
                    new_qty = st.number_input("New Quantity", value=float(edit_row['quantity']), min_value=0.1, step=1.0, key="edit_qty")
                    new_avg = st.number_input("New Avg Price (₹)", value=float(edit_row['avg_price']), min_value=0.01, step=0.01, key="edit_avg")
                    
                    if st.button("💾 Save Changes", type="primary", key="save_edit"):
                        db.update_holding(edit_id, new_qty, new_avg)
                        st.success(f"✅ {edit_row['symbol']} updated!")
                        time.sleep(1)
                        st.rerun()
                
                st.divider()
                st.subheader("🗑️ Delete Holding")
                
                delete_options = ["-- Select --"] + [f"{row['symbol']} ({row['id']})" for _, row in holdings.iterrows()]
                delete_selection = st.selectbox("Choose holding to delete", delete_options, key="delete_select")
                
                if delete_selection != "-- Select --":
                    delete_id = int(delete_selection.split('(')[1].strip(')'))
                    delete_row = holdings[holdings['id'] == delete_id].iloc[0]
                    
                    st.warning(f"⚠️ Delete {delete_row['symbol']}?")
                    st.caption(f"{delete_row['quantity']:.0f} shares @ ₹{delete_row['avg_price']:.2f}")
                    
                    if st.button("✅ Yes, Delete", type="primary", key="confirm_delete"):
                        db.delete_holding(delete_id)
                        st.success(f"✅ {delete_row['symbol']} deleted!")
                        time.sleep(1)
                        st.rerun()
            
            # Holdings display
            st.markdown("### 📋 Holdings Details")
            
            for idx in range(0, len(holdings), 3):
                cols = st.columns(3)
                
                for col_idx, col in enumerate(cols):
                    if idx + col_idx < len(holdings):
                        row = holdings.iloc[idx + col_idx]
                        
                        with col:
                            try:
                                ticker = yf.Ticker(f"{row['symbol']}.NS")
                                #current_price = ticker.history(period='1d')['Close'].iloc[-1]
                                current_price = get_current_price(row['symbol'])
                                if current_price == 0:
                                    st.warning(f"⚠️ Cannot fetch price for {row['symbol']}")
                                    continue
                                current_value = current_price * row['quantity']
                                pnl = current_value - row['invested_amount']
                                pnl_pct = (pnl / row['invested_amount'] * 100)
                                
                                with st.container():
                                    # Show SHORT indicator if negative quantity
                                    position_type = "📉 SHORT" if row['quantity'] < 0 else "📈 LONG"
                                    st.markdown(f"### {row['symbol']} {position_type}")
                                    st.caption(f"{row['company_name'][:25]}")
                                    
                                    qty_display = abs(row['quantity'])  # Show absolute value
                                    
                                    st.write(f"**Qty:** {row['quantity']:.0f} | **Avg:** ₹{row['avg_price']:.2f}")
                                    st.write(f"**CMP:** ₹{current_price:.2f}")
                                    
                                    pnl_class = "profit" if pnl >= 0 else "loss"
                                    st.markdown(f"<p class='{pnl_class}'>P&L: {format_currency(pnl)} ({pnl_pct:+.2f}%)</p>", unsafe_allow_html=True)
                                    
                                    st.caption(f"ID: {row['id']}")
                                    st.divider()
                            
                            except Exception as e:
                                st.error(f"Error loading {row['symbol']}: {str(e)}")
            
            # Auto-refresh
            if 'auto_refresh' in locals() and auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        
        else:
            st.info("📤 No holdings yet. Go to 'Upload Files' to import your portfolio!")
    
    # ==================== ADD TRANSACTION ====================
    elif page == "➕ Add Transaction":
        st.header("Add Transaction")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                trans_type = st.selectbox("Type", ["BUY", "SELL"])
                symbol = st.text_input("Symbol (e.g., RELIANCE)", "").upper()
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            
            with col2:
                trans_date = st.date_input("Date", value=date.today())
                price = st.number_input("Price (₹)", min_value=0.01, value=100.0, step=0.01)
                notes = st.text_area("Notes (optional)", "")
            
            if symbol:
                instrument_type, yahoo_symbol = detect_instrument_type(symbol)
                current_price = get_current_price(symbol)
                
                if current_price > 0:
                    st.success(f"✅ {symbol} ({instrument_type.upper()}) - Current: ₹{current_price:.2f}")
                    company_name = symbol
                else:
                    st.info(f"ℹ️ {symbol} ({instrument_type.upper()}) - Manual price entry required")
                    company_name = symbol
            else:
                company_name = ""
            
            total = quantity * price
            st.info(f"💰 Total Amount: {format_currency(total)}")
            
            submitted = st.form_submit_button("✅ Add Transaction", type="primary")
            
            if submitted and symbol and company_name:
                try:
                    db.add_transaction(symbol, company_name, trans_type, quantity, price, trans_date, notes)
                    st.success("✅ Transaction added successfully!")
                    # BALLOON ANIMATION REMOVED
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ==================== TRANSACTION HISTORY ====================
    elif page == "📜 Transaction History":
        st.header("Transaction History")
        
        transactions = db.get_transactions(limit=100)
        
        if not transactions.empty:
            st.dataframe(
                transactions[['transaction_date', 'symbol', 'transaction_type',
                             'quantity', 'price', 'total_amount', 'notes']],
                use_container_width=True,
                height=400
            )
            
            st.info(f"📊 Showing last {len(transactions)} transactions")
        else:
            st.info("📋 No transactions yet.")
    
    # ==================== REALIZED P&L ====================
    elif page == "💰 Realized P&L":
        st.header("Realized Profit & Loss")
        
        realized = db.get_realized_pnl()
        
        if not realized.empty:
            total = realized['profit_loss'].sum()
            avg_return = realized['profit_loss_pct'].mean()
            winners = len(realized[realized['profit_loss'] > 0])
            losers = len(realized[realized['profit_loss'] < 0])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Realized P&L", format_currency(total))
            with col2:
                st.metric("Avg Return %", f"{avg_return:.2f}%")
            with col3:
                st.metric("Winning Trades", winners)
            with col4:
                st.metric("Losing Trades", losers)
            
            st.divider()
            
            st.dataframe(
                realized[['symbol', 'company_name', 'quantity', 'buy_price', 'sell_price',
                         'profit_loss', 'profit_loss_pct']],
                use_container_width=True,
                height=400
            )
        else:
            st.info("💰 No realized P&L yet. Sell some holdings to see booked profits/losses.")
    
    # ==================== UPLOAD FILES ====================
    elif page == "📤 Upload Files":
        st.header("Upload Portfolio & Stock List")
        
        tab1, tab2 = st.tabs(["📊 Portfolio Import", "📋 Stock List"])
        
        with tab1:
            st.subheader("Import Portfolio Data")
            
            # Choose import type
            import_type = st.radio(
                "Import Type",
                ["📊 Holdings Only (Current Portfolio)", "📜 Transaction History (Full Reconstruction)"],
                horizontal=True
            )
            
            if import_type == "📊 Holdings Only (Current Portfolio)":
                st.info("📋 Upload file with columns: **Symbol, Company, Quantity, Avg Price**")
                st.caption("This will replace your current holdings with the uploaded data.")
                
                # Show export option
                holdings = db.get_holdings()
                if not holdings.empty:
                    st.markdown("#### 📥 Export Current Portfolio")
                    export_df = holdings[['symbol', 'company_name', 'quantity', 'avg_price', 'invested_amount']]
                    export_df.columns = ['Symbol', 'Company', 'Quantity', 'Avg Price', 'Invested Amount']
                    
                    # CSV export
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        csv_export,
                        f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
                    # Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Portfolio')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        "📥 Download as Excel",
                        excel_data,
                        f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.divider()
                
                # Import Holdings
                st.markdown("#### 📤 Import Holdings")
                portfolio_file = st.file_uploader(
                    "Upload Portfolio File",
                    type=['xlsx', 'xls', 'csv', 'txt'],
                    key='portfolio_file'
                )
                
                if portfolio_file is not None:
                    try:
                        preview_df = load_portfolio_from_file(portfolio_file)
                        
                        if not preview_df.empty:
                            st.write("**Preview:**")
                            st.dataframe(preview_df.head(), use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("✅ Import Portfolio", type="primary"):
                                    success_count, errors = db.bulk_import_portfolio(preview_df)
                                    
                                    if success_count > 0:
                                        st.success(f"✅ Imported {success_count} holdings!")
                                    
                                    if errors:
                                        st.warning(f"⚠️ {len(errors)} errors:")
                                        for error in errors[:5]:
                                            st.text(error)
                                    
                                    st.rerun()
                            
                            with col2:
                                st.warning("⚠️ This will replace existing holdings!")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            else:  # Transaction History Import
                st.info("📋 Upload file with columns: **Symbol, Company, Type (BUY/SELL), Quantity, Price, Date, Notes**")
                st.caption("This will import all transactions and automatically build your portfolio.")
                
                # Show sample format
                with st.expander("📋 View Sample Format"):
                    sample_df = pd.DataFrame({
                        'Symbol': ['RELIANCE', 'RELIANCE', 'TCS', 'TCS'],
                        'Company': ['Reliance Industries', 'Reliance Industries', 'TCS Ltd', 'TCS Ltd'],
                        'Type': ['BUY', 'SELL', 'BUY', 'BUY'],
                        'Quantity': [100, 50, 200, 100],
                        'Price': [2000.00, 2100.00, 3500.00, 3600.00],
                        'Date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-01'],
                        'Notes': ['Initial buy', 'Partial profit booking', 'Long term', 'Averaging']
                    })
                    st.dataframe(sample_df, use_container_width=True)
                
                # Export transactions
                transactions = db.get_transactions(limit=1000)
                if not transactions.empty:
                    st.markdown("#### 📥 Export Transaction History")
                    export_df = transactions[['symbol', 'company_name', 'transaction_type', 'quantity', 
                                             'price', 'transaction_date', 'notes']]
                    export_df.columns = ['Symbol', 'Company', 'Type', 'Quantity', 'Price', 'Date', 'Notes']
                    
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Transactions as CSV",
                        csv_export,
                        f"transactions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
                    st.divider()
                
                # Import Transactions
                st.markdown("#### 📤 Import Transactions")
                transaction_file = st.file_uploader(
                    "Upload Transaction File",
                    type=['xlsx', 'xls', 'csv', 'txt'],
                    key='transaction_file'
                )
                
                if transaction_file is not None:
                    try:
                        preview_df = load_transactions_from_file(transaction_file)
                        
                        if not preview_df.empty:
                            st.write("**Preview:**")
                            st.dataframe(preview_df.head(10), use_container_width=True)
                            st.caption(f"Total: {len(preview_df)} transactions")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                clear_existing = st.checkbox("Clear existing data before import", value=True)
                            
                            with col2:
                                if st.button("✅ Import Transactions", type="primary"):
                                    if clear_existing:
                                        # Clear existing data
                                        conn = sqlite3.connect("trading_system.db")
                                        conn.execute("DELETE FROM holdings")
                                        conn.execute("DELETE FROM transactions")
                                        conn.execute("DELETE FROM realized_pnl")
                                        conn.commit()
                                        conn.close()
                                    
                                    success_count, errors = db.bulk_import_transactions(preview_df)
                                    
                                    if success_count > 0:
                                        st.success(f"✅ Imported {success_count} transactions!")
                                    
                                    if errors:
                                        st.warning(f"⚠️ {len(errors)} errors:")
                                        for error in errors[:10]:
                                            st.text(error)
                                    
                                    st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
        
        with tab2:
            st.subheader("Upload Stock List for Scanner")
            st.info("📋 Upload Excel/CSV/TXT file with stock symbols")
            
            if st.session_state.stock_list:
                st.success(f"✅ Currently using {len(st.session_state.stock_list)} symbols from GitHub")
                
                with st.expander("View Current Symbols"):
                    st.write(", ".join(st.session_state.stock_list[:50]))
                    if len(st.session_state.stock_list) > 50:
                        st.caption(f"... and {len(st.session_state.stock_list) - 50} more")
            
            stock_file = st.file_uploader(
                "Upload Stock Symbols",
                type=['xlsx', 'xls', 'csv', 'txt'],
                key='stock_file'
            )
            
            if stock_file is not None:
                try:
                    symbols = load_stock_list_from_file(stock_file)
                    
                    if symbols:
                        st.write(f"**Found {len(symbols)} symbols:**")
                        st.write(", ".join(symbols[:20]))
                        if len(symbols) > 20:
                            st.caption(f"... and {len(symbols) - 20} more")
                        
                        if st.button("💾 Save to GitHub", type="primary"):
                            if db.github.configured:
                                content = '\n'.join(symbols)
                                if db.github.save_to_github('stock_symbols.txt', content, 'Updated stock list'):
                                    st.session_state.stock_list = symbols
                                    st.success("✅ Stock list saved to GitHub!")
                                    st.rerun()
                            else:
                                st.error("❌ Please configure GitHub first!")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    ✨ **Stock Scanner:**
    - Strategy: 9/21 SMA ✅
    - News Integration ✅
    - Volume Analysis ✅
    - Portfolio Manager ✅
    """)
    st.sidebar.caption("Stock Scanner - By Ashish Gupta")


if __name__ == "__main__":
    main()
