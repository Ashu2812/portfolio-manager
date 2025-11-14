"""
UNIFIED TRADING SYSTEM v2.2 ENHANCED
NEW FEATURES:
- Unrealised P&L at top of Portfolio Manager
- CSV portfolio upload/re-import
- GitHub storage for portfolio & stock list persistence
- Balloon animation removed
- Base SMA & Volume strategy 100% intact
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

# Page config
st.set_page_config(
    page_title="Trading System v2.2",
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
    .pnl-metric {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== GITHUB STORAGE MANAGER ====================

class GitHubStorageManager:
    """Manages persistent storage on GitHub for portfolio and stock list"""
    
    def __init__(self):
        self.api_base = "https://api.github.com"
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
            # First, try to get the file SHA (if it exists)
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/contents/{filename}"
            headers = {
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers)
            sha = response.json().get('sha') if response.status_code == 200 else None
            
            # Encode content to base64
            content_bytes = content.encode('utf-8')
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            
            # Create or update file
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
            st.error(f"GitHub load error: {str(e)}")
            return None
    
    def delete_from_github(self, filename: str, commit_message: str) -> bool:
        """Delete file from GitHub repository"""
        if not self.configured:
            return False
        
        try:
            # Get the file SHA first
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/contents/{filename}"
            headers = {
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                sha = response.json()['sha']
                
                # Delete the file
                data = {
                    'message': commit_message,
                    'sha': sha,
                    'branch': 'main'
                }
                
                response = requests.delete(url, headers=headers, json=data)
                return response.status_code == 200
            
            return False
            
        except Exception as e:
            st.error(f"GitHub delete error: {str(e)}")
            return False


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
        
        # Fetch from all sources
        all_articles.extend(self.fetch_newsapi(symbol, company_name))
        all_articles.extend(self.fetch_google_news(symbol, company_name))
        all_articles.extend(self.fetch_economic_times(symbol, company_name))
        all_articles.extend(self.fetch_moneycontrol(symbol, company_name))
        
        # Remove duplicates based on title similarity
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title_lower = article['title'].lower()[:50]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return unique_articles[:max_total]


# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Enhanced database manager with GitHub sync capabilities"""
    
    def __init__(self, db_path="trading_system.db"):
        self.db_path = db_path
        self.github = GitHubStorageManager()
        self.init_database()
        
        # Try to load portfolio from GitHub on init
        if self.github.configured:
            self.sync_from_github()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            """)
            
            conn.execute("""
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
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS realized_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    buy_price REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    profit_loss_pct REAL NOT NULL,
                    sell_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def sync_to_github(self):
        """Sync portfolio data to GitHub"""
        if not self.github.configured:
            return False
        
        try:
            # Export portfolio as JSON
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
            
            if success:
                st.success("✅ Portfolio synced to GitHub!")
            
            return success
            
        except Exception as e:
            st.error(f"Sync error: {str(e)}")
            return False
    
    def sync_from_github(self):
        """Load portfolio data from GitHub"""
        if not self.github.configured:
            return False
        
        try:
            content = self.github.load_from_github('portfolio_data.json')
            
            if content:
                portfolio_data = json.loads(content)
                
                # Clear existing data
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM holdings")
                    conn.execute("DELETE FROM transactions")
                    conn.execute("DELETE FROM realized_pnl")
                    conn.commit()
                
                # Import holdings
                if portfolio_data.get('holdings'):
                    holdings_df = pd.DataFrame(portfolio_data['holdings'])
                    holdings_df.to_sql('holdings', conn, if_exists='append', index=False)
                
                # Import transactions
                if portfolio_data.get('transactions'):
                    trans_df = pd.DataFrame(portfolio_data['transactions'])
                    trans_df.to_sql('transactions', conn, if_exists='append', index=False)
                
                # Import realized P&L
                if portfolio_data.get('realized_pnl'):
                    realized_df = pd.DataFrame(portfolio_data['realized_pnl'])
                    realized_df.to_sql('realized_pnl', conn, if_exists='append', index=False)
                
                st.success("✅ Portfolio loaded from GitHub!")
                return True
                
        except Exception as e:
            st.warning(f"No portfolio found on GitHub or load error: {str(e)}")
            return False
    
    def add_transaction(self, symbol: str, company_name: str, trans_type: str,
                       quantity: float, price: float, trans_date: date, notes: str = ""):
        """Add transaction and update holdings"""
        total = quantity * price
        
        with sqlite3.connect(self.db_path) as conn:
            # Add transaction
            conn.execute("""
                INSERT INTO transactions 
                (symbol, company_name, transaction_type, quantity, price, total_amount, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, company_name, trans_type, quantity, price, total, trans_date, notes))
            
            # Update holdings
            if trans_type == "BUY":
                existing = conn.execute(
                    "SELECT quantity, avg_price FROM holdings WHERE symbol = ?", (symbol,)
                ).fetchone()
                
                if existing:
                    old_qty, old_avg = existing
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
                    new_invested = new_qty * new_avg
                    
                    conn.execute("""
                        UPDATE holdings 
                        SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    """, (new_qty, new_avg, new_invested, symbol))
                else:
                    conn.execute("""
                        INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, company_name, quantity, price, total))
            
            elif trans_type == "SELL":
                existing = conn.execute(
                    "SELECT quantity, avg_price FROM holdings WHERE symbol = ?", (symbol,)
                ).fetchone()
                
                if existing:
                    old_qty, old_avg = existing
                    
                    # Calculate realized P&L
                    profit_loss = (price - old_avg) * quantity
                    profit_loss_pct = ((price - old_avg) / old_avg) * 100
                    
                    # Record realized P&L
                    conn.execute("""
                        INSERT INTO realized_pnl 
                        (symbol, company_name, quantity, buy_price, sell_price, profit_loss, profit_loss_pct, sell_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, company_name, quantity, old_avg, price, profit_loss, profit_loss_pct, trans_date))
                    
                    # Update or remove holding
                    new_qty = old_qty - quantity
                    if new_qty > 0:
                        new_invested = new_qty * old_avg
                        conn.execute("""
                            UPDATE holdings 
                            SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE symbol = ?
                        """, (new_qty, new_invested, symbol))
                    else:
                        conn.execute("DELETE FROM holdings WHERE symbol = ?", (symbol,))
            
            conn.commit()
        
        # Auto-sync to GitHub after transaction
        if self.github.configured:
            self.sync_to_github()
    
    def get_holdings(self) -> pd.DataFrame:
        """Get all holdings"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM holdings ORDER BY symbol", conn)
        return df
    
    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        """Get transaction history"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM transactions ORDER BY transaction_date DESC, created_at DESC LIMIT {limit}",
                conn
            )
        return df
    
    def get_realized_pnl(self) -> pd.DataFrame:
        """Get realized P&L"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM realized_pnl ORDER BY sell_date DESC", conn)
        return df
    
    def update_holding(self, holding_id: int, quantity: float, avg_price: float):
        """Update holding"""
        with sqlite3.connect(self.db_path) as conn:
            invested = quantity * avg_price
            conn.execute("""
                UPDATE holdings 
                SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (quantity, avg_price, invested, holding_id))
            conn.commit()
        
        # Auto-sync to GitHub
        if self.github.configured:
            self.sync_to_github()
    
    def delete_holding(self, holding_id: int):
        """Delete holding"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
            conn.commit()
        
        # Auto-sync to GitHub
        if self.github.configured:
            self.sync_to_github()
    
    def import_portfolio_csv(self, csv_content: str):
        """Import portfolio from CSV"""
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Expected columns: symbol, company_name, quantity, avg_price, invested_amount
            required_cols = ['symbol', 'company_name', 'quantity', 'avg_price', 'invested_amount']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing holdings
                conn.execute("DELETE FROM holdings")
                
                # Import new holdings
                for _, row in df.iterrows():
                    conn.execute("""
                        INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                        VALUES (?, ?, ?, ?, ?)
                    """, (row['symbol'], row['company_name'], row['quantity'], 
                         row['avg_price'], row['invested_amount']))
                
                conn.commit()
            
            # Auto-sync to GitHub
            if self.github.configured:
                self.sync_to_github()
            
            return True, f"Imported {len(df)} holdings successfully!"
            
        except Exception as e:
            return False, f"Import error: {str(e)}"
    
    def clear_all_data(self):
        """Clear all portfolio data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM holdings")
            conn.execute("DELETE FROM transactions")
            conn.execute("DELETE FROM realized_pnl")
            conn.commit()
        
        # Delete from GitHub if configured
        if self.github.configured:
            self.github.delete_from_github(
                'portfolio_data.json',
                'Cleared all portfolio data'
            )


# ==================== STOCK SCANNER ====================

class StockScanner:
    """Stock scanner with SMA & Volume strategy (UNCHANGED)"""
    
    def __init__(self, symbols_file: str = None):
        self.symbols_file = symbols_file
        self.github = GitHubStorageManager()
        
        # Try to load stock list from GitHub
        if self.github.configured and not symbols_file:
            self.load_symbols_from_github()
    
    def load_symbols_from_github(self):
        """Load stock list from GitHub"""
        try:
            content = self.github.load_from_github('stock_symbols.txt')
            if content:
                symbols = [line.strip() for line in content.split('\n') if line.strip()]
                if symbols:
                    # Save to session state
                    st.session_state.stock_symbols_github = symbols
                    st.success(f"✅ Loaded {len(symbols)} symbols from GitHub!")
        except Exception as e:
            st.warning(f"Could not load symbols from GitHub: {str(e)}")
    
    def save_symbols_to_github(self, symbols: List[str]):
        """Save stock list to GitHub"""
        if not self.github.configured:
            return False
        
        try:
            content = '\n'.join(symbols)
            success = self.github.save_to_github(
                'stock_symbols.txt',
                content,
                f'Updated stock list - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            
            if success:
                st.session_state.stock_symbols_github = symbols
                st.success("✅ Stock list saved to GitHub!")
            
            return success
            
        except Exception as e:
            st.error(f"Error saving symbols: {str(e)}")
            return False
    
    def load_symbols(self) -> List[str]:
        """Load symbols from file or GitHub"""
        # First check GitHub session state
        if 'stock_symbols_github' in st.session_state:
            return st.session_state.stock_symbols_github
        
        # Then check uploaded file
        if self.symbols_file:
            try:
                content = self.symbols_file.read().decode('utf-8')
                symbols = [line.strip() for line in content.split('\n') if line.strip()]
                
                # Save to GitHub for persistence
                if self.github.configured:
                    self.save_symbols_to_github(symbols)
                
                return symbols
            except:
                st.error("Error reading symbols file")
                return []
        
        return []
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average (UNCHANGED)"""
        return data['Close'].rolling(window=period).mean()
    
    def calculate_volume_avg(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate average volume (UNCHANGED)"""
        return data['Volume'].tail(period).mean()
    
    def scan_stock(self, symbol: str, sma_short: int = 20, sma_long: int = 50,
                   volume_multiplier: float = 1.5) -> Dict:
        """
        Scan single stock using SMA & Volume strategy (100% UNCHANGED)
        
        Strategy Rules:
        1. Short SMA crosses above Long SMA (Golden Cross)
        2. Current volume > (Average volume * multiplier)
        3. Price trending up
        """
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period='3mo')
            
            if len(data) < sma_long:
                return {'symbol': symbol, 'signal': 'INSUFFICIENT_DATA', 'valid': False}
            
            # Calculate SMAs
            data['SMA_Short'] = self.calculate_sma(data, sma_short)
            data['SMA_Long'] = self.calculate_sma(data, sma_long)
            
            # Get latest values
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            current_price = latest['Close']
            sma_short_current = latest['SMA_Short']
            sma_long_current = latest['SMA_Long']
            current_volume = latest['Volume']
            
            # Calculate average volume
            avg_volume = self.calculate_volume_avg(data)
            
            # Strategy Signals
            golden_cross = (sma_short_current > sma_long_current and 
                          previous['SMA_Short'] <= previous['SMA_Long'])
            
            above_sma = sma_short_current > sma_long_current
            volume_spike = current_volume > (avg_volume * volume_multiplier)
            price_trend = current_price > data['Close'].iloc[-5]
            
            # Determine signal
            if golden_cross and volume_spike:
                signal = 'STRONG_BUY'
            elif above_sma and volume_spike:
                signal = 'BUY'
            elif above_sma and price_trend:
                signal = 'HOLD'
            else:
                signal = 'NO_SIGNAL'
            
            # Get company info
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'current_price': round(current_price, 2),
                'sma_short': round(sma_short_current, 2),
                'sma_long': round(sma_long_current, 2),
                'volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'volume_ratio': round(current_volume / avg_volume, 2),
                'signal': signal,
                'golden_cross': golden_cross,
                'above_sma': above_sma,
                'volume_spike': volume_spike,
                'price_trend': price_trend,
                'valid': True
            }
            
        except Exception as e:
            return {'symbol': symbol, 'signal': 'ERROR', 'error': str(e), 'valid': False}
    
    def scan_multiple(self, symbols: List[str], sma_short: int = 20,
                     sma_long: int = 50, volume_multiplier: float = 1.5,
                     progress_callback=None) -> pd.DataFrame:
        """Scan multiple stocks (UNCHANGED)"""
        results = []
        total = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(idx + 1, total, symbol)
            
            result = self.scan_stock(symbol, sma_short, sma_long, volume_multiplier)
            if result['valid']:
                results.append(result)
            
            time.sleep(0.5)  # Rate limiting
        
        if results:
            df = pd.DataFrame(results)
            df = df[df['signal'].isin(['STRONG_BUY', 'BUY', 'HOLD'])]
            df = df.sort_values('volume_ratio', ascending=False)
            return df
        
        return pd.DataFrame()


# ==================== UTILITY FUNCTIONS ====================

def format_currency(amount: float) -> str:
    """Format currency in Indian style"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:,.2f}"


def get_stock_info(symbol: str) -> Dict:
    """Get current stock information"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period='1d')
        
        return {
            'valid': True,
            'name': info.get('longName', symbol),
            'current_price': hist['Close'].iloc[-1] if not hist.empty else 0,
            'change': info.get('regularMarketChange', 0),
            'change_pct': info.get('regularMarketChangePercent', 0)
        }
    except:
        return {'valid': False, 'name': symbol, 'current_price': 0}


# ==================== MAIN APPLICATION ====================

def main():
    # Initialize database
    db = DatabaseManager()
    
    # Sidebar
    st.sidebar.title("📊 Trading System v2.2")
    
    # GitHub Configuration Section
    with st.sidebar.expander("☁️ GitHub Storage Setup", expanded=False):
        st.markdown("**Configure GitHub for persistent storage**")
        
        github_owner = st.text_input("GitHub Username", 
                                     value=st.session_state.get('github_owner', ''),
                                     help="Your GitHub username")
        github_repo = st.text_input("Repository Name", 
                                    value=st.session_state.get('github_repo', ''),
                                    help="e.g., trading-data")
        github_token = st.text_input("Personal Access Token", 
                                     value=st.session_state.get('github_token', ''),
                                     type="password",
                                     help="GitHub PAT with repo access")
        
        if st.button("💾 Save GitHub Config"):
            if github_owner and github_repo and github_token:
                st.session_state.github_owner = github_owner
                st.session_state.github_repo = github_repo
                st.session_state.github_token = github_token
                db.github.load_config()
                st.success("✅ GitHub configured!")
                st.rerun()
            else:
                st.error("Please fill all fields")
        
        if db.github.configured:
            st.success("✅ GitHub Connected")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬆️ Sync to GitHub"):
                    db.sync_to_github()
            with col2:
                if st.button("⬇️ Load from GitHub"):
                    db.sync_from_github()
                    st.rerun()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📈 Stock Scanner", "💼 Portfolio Manager", "➕ Add Transaction",
         "📜 Transaction History", "💰 Realized P&L", "📤 Upload Files", 
         "🗑️ Clear Data"]
    )
    
    st.sidebar.divider()
    
    # Main content
    st.markdown("<h1 class='main-header'>🚀 Unified Trading System v2.2</h1>", unsafe_allow_html=True)
    
    # ==================== STOCK SCANNER ====================
    if page == "📈 Stock Scanner":
        st.header("Stock Scanner - SMA & Volume Strategy")
        
        st.info("📋 Upload a text file with NSE symbols (one per line) or load from GitHub")
        
        # Check for GitHub symbols first
        has_github_symbols = 'stock_symbols_github' in st.session_state
        
        if has_github_symbols:
            st.success(f"✅ Using {len(st.session_state.stock_symbols_github)} symbols from GitHub")
            use_github = st.checkbox("Use symbols from GitHub", value=True)
            
            if not use_github:
                symbols_file = st.file_uploader("Upload Symbols File", type=['txt'])
            else:
                symbols_file = None
        else:
            symbols_file = st.file_uploader("Upload Symbols File", type=['txt'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sma_short = st.number_input("Short SMA Period", min_value=5, max_value=50, value=20)
        with col2:
            sma_long = st.number_input("Long SMA Period", min_value=20, max_value=200, value=50)
        with col3:
            volume_multiplier = st.number_input("Volume Multiplier", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("🔍 Start Scanning", type="primary"):
            scanner = StockScanner(symbols_file)
            symbols = scanner.load_symbols()
            
            if not symbols:
                st.error("❌ No symbols to scan. Please upload a file or load from GitHub.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, symbol):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning {symbol}... ({current}/{total})")
                
                results_df = scanner.scan_multiple(
                    symbols, sma_short, sma_long, volume_multiplier, update_progress
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if not results_df.empty:
                    st.success(f"✅ Found {len(results_df)} trading opportunities!")
                    
                    # Display results
                    st.dataframe(
                        results_df[['symbol', 'company_name', 'current_price', 
                                  'signal', 'volume_ratio', 'sma_short', 'sma_long']],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                else:
                    st.warning("⚠️ No trading signals found.")
    
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
            # ==================== UNREALISED P&L SUMMARY (NEW FEATURE) ====================
            st.markdown("### 📊 Portfolio Summary")
            
            # Calculate unrealised P&L for all holdings
            total_invested = 0
            total_current_value = 0
            portfolio_data = []
            
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    unrealised_pnl = current_value - row['invested_amount']
                    unrealised_pnl_pct = (unrealised_pnl / row['invested_amount']) * 100
                    
                    total_invested += row['invested_amount']
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
            
            # Display summary metrics
            if portfolio_data:
                total_unrealised_pnl = total_current_value - total_invested
                total_unrealised_pnl_pct = (total_unrealised_pnl / total_invested * 100) if total_invested > 0 else 0
                
                # Summary card
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
            
            # Sidebar for Edit/Delete operations
            with st.sidebar:
                st.divider()
                st.subheader("✏️ Edit Holding")
                
                # Select holding to edit
                edit_options = ["-- Select --"] + [f"{row['symbol']} ({row['id']})" for _, row in holdings.iterrows()]
                edit_selection = st.selectbox("Choose holding to edit", edit_options, key="edit_select")
                
                if edit_selection != "-- Select --":
                    edit_id = int(edit_selection.split('(')[1].strip(')'))
                    edit_row = holdings[holdings['id'] == edit_id].iloc[0]
                    
                    st.write(f"**Editing: {edit_row['symbol']}**")
                    new_qty = st.number_input("New Quantity", value=float(edit_row['quantity']), min_value=0.1, step=1.0, key="edit_qty_input")
                    new_avg = st.number_input("New Avg Price (₹)", value=float(edit_row['avg_price']), min_value=0.01, step=0.01, key="edit_avg_input")
                    
                    if st.button("💾 Save Changes", type="primary", key="save_edit_btn"):
                        db.update_holding(edit_id, new_qty, new_avg)
                        st.success(f"✅ {edit_row['symbol']} updated!")
                        time.sleep(1)
                        st.rerun()
                
                st.divider()
                st.subheader("🗑️ Delete Holding")
                
                # Select holding to delete
                delete_options = ["-- Select --"] + [f"{row['symbol']} ({row['id']})" for _, row in holdings.iterrows()]
                delete_selection = st.selectbox("Choose holding to delete", delete_options, key="delete_select")
                
                if delete_selection != "-- Select --":
                    delete_id = int(delete_selection.split('(')[1].strip(')'))
                    delete_row = holdings[holdings['id'] == delete_id].iloc[0]
                    
                    st.warning(f"⚠️ Delete {delete_row['symbol']}?")
                    st.caption(f"{delete_row['quantity']:.0f} shares @ ₹{delete_row['avg_price']:.2f}")
                    
                    if st.button("✅ Yes, Delete", type="primary", key="confirm_delete_btn"):
                        db.delete_holding(delete_id)
                        st.success(f"✅ {delete_row['symbol']} deleted!")
                        time.sleep(1)
                        st.rerun()
            
            # Display holdings in tiles
            st.markdown("### 📋 Holdings Details")
            
            # Create tiles in rows of 3
            for idx in range(0, len(holdings), 3):
                cols = st.columns(3)
                
                for col_idx, col in enumerate(cols):
                    if idx + col_idx < len(holdings):
                        row = holdings.iloc[idx + col_idx]
                        
                        with col:
                            try:
                                ticker = yf.Ticker(f"{row['symbol']}.NS")
                                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                                current_value = current_price * row['quantity']
                                pnl = current_value - row['invested_amount']
                                pnl_pct = (pnl / row['invested_amount'] * 100)
                                
                                # Tile container
                                with st.container():
                                    st.markdown(f"### {row['symbol']}")
                                    st.caption(f"{row['company_name'][:25]}")
                                    
                                    st.write(f"**Qty:** {row['quantity']:.0f} | **Avg:** ₹{row['avg_price']:.2f}")
                                    st.write(f"**CMP:** ₹{current_price:.2f}")
                                    
                                    pnl_class = "profit" if pnl >= 0 else "loss"
                                    st.markdown(f"<p class='{pnl_class}'>P&L: {format_currency(pnl)} ({pnl_pct:+.2f}%)</p>", unsafe_allow_html=True)
                                    
                                    st.caption(f"ID: {row['id']}")
                                    st.divider()
                            
                            except Exception as e:
                                st.error(f"Error loading {row['symbol']}: {str(e)}")
            
            # Auto-refresh logic
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
                stock_info = get_stock_info(symbol)
                if stock_info['valid']:
                    st.success(f"✅ {stock_info['name']} - Current: ₹{stock_info['current_price']:.2f}")
                    company_name = stock_info['name']
                else:
                    st.warning("⚠️ Could not fetch stock info. Using symbol as name.")
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
                    # BALLOON ANIMATION REMOVED (as requested)
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
    
    # ==================== UPLOAD FILES (NEW CSV IMPORT FEATURE) ====================
    elif page == "📤 Upload Files":
        st.header("Upload Portfolio & Stock List")
        
        tab1, tab2 = st.tabs(["📊 Portfolio Import", "📋 Stock List"])
        
        with tab1:
            st.subheader("Import Portfolio from CSV")
            st.info("📋 Upload a CSV file with columns: symbol, company_name, quantity, avg_price, invested_amount")
            
            # Show export option first
            holdings = db.get_holdings()
            if not holdings.empty:
                st.markdown("#### 📥 Export Current Portfolio")
                export_df = holdings[['symbol', 'company_name', 'quantity', 'avg_price', 'invested_amount']]
                csv_export = export_df.to_csv(index=False)
                
                st.download_button(
                    "📥 Download Current Portfolio as CSV",
                    csv_export,
                    f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    help="Download your portfolio in CSV format for backup or re-import"
                )
                
                st.divider()
            
            # CSV Upload
            st.markdown("#### 📤 Import Portfolio from CSV")
            portfolio_csv = st.file_uploader("Upload Portfolio CSV", type=['csv'], key='portfolio_csv')
            
            if portfolio_csv is not None:
                try:
                    # Preview the uploaded file
                    preview_df = pd.read_csv(portfolio_csv)
                    st.write("**Preview:**")
                    st.dataframe(preview_df.head(), use_container_width=True)
                    
                    # Reset file pointer
                    portfolio_csv.seek(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Import Portfolio", type="primary"):
                            csv_content = portfolio_csv.read().decode('utf-8')
                            success, message = db.import_portfolio_csv(csv_content)
                            
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col2:
                        st.warning("⚠️ This will replace existing portfolio!")
                
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        with tab2:
            st.subheader("Upload Stock List for Scanner")
            st.info("📋 Upload a text file with NSE symbols (one per line)")
            
            # Show current GitHub symbols if available
            if 'stock_symbols_github' in st.session_state:
                st.success(f"✅ Currently using {len(st.session_state.stock_symbols_github)} symbols from GitHub")
                
                with st.expander("View Current Symbols"):
                    st.write(", ".join(st.session_state.stock_symbols_github[:50]))
                    if len(st.session_state.stock_symbols_github) > 50:
                        st.caption(f"... and {len(st.session_state.stock_symbols_github) - 50} more")
            
            # Upload new stock list
            stock_file = st.file_uploader("Upload Stock Symbols", type=['txt'], key='stock_file')
            
            if stock_file is not None:
                try:
                    content = stock_file.read().decode('utf-8')
                    symbols = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    st.write(f"**Found {len(symbols)} symbols:**")
                    st.write(", ".join(symbols[:20]))
                    if len(symbols) > 20:
                        st.caption(f"... and {len(symbols) - 20} more")
                    
                    if st.button("💾 Save to GitHub", type="primary"):
                        if db.github.configured:
                            scanner = StockScanner()
                            if scanner.save_symbols_to_github(symbols):
                                st.rerun()
                        else:
                            st.error("❌ Please configure GitHub in the sidebar first!")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    # ==================== CLEAR DATA ====================
    elif page == "🗑️ Clear Data":
        st.header("Clear All Data")
        
        st.warning("⚠️ **WARNING:** This will permanently delete all your data!")
        
        st.markdown("""
        This action will delete:
        - All portfolio holdings
        - All transaction history
        - All realized P&L records
        - Data from GitHub (if configured)
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            confirm_text = st.text_input("Type 'DELETE' to confirm", "")
            
            if confirm_text == "DELETE":
                if st.button("🗑️ Delete Everything", type="primary"):
                    db.clear_all_data()
                    st.success("✅ All data cleared!")
                    st.info("Please refresh the page")
            else:
                st.button("🗑️ Delete Everything", disabled=True)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    ✨ **v2.2 ENHANCED:**
    - Unrealised P&L Display ✅
    - CSV Portfolio Import ✅
    - GitHub Persistence ✅
    - No Balloon Animation ✅
    - Base Strategy Intact ✅
    """)
    st.sidebar.caption("Stock Scanner - By Ashish Gupta")


if __name__ == "__main__":
    main()
