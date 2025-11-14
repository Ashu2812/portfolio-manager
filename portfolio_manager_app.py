"""
UNIFIED TRADING SYSTEM v3.0 ENHANCED
New Features:
- Unrealized P&L at the top of Portfolio Manager
- Portfolio persistence using GitHub
- Stock list saved on GitHub (accessible from laptop & mobile)
- CSV upload for portfolio
- Balloon animation removed
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
import sqlite3
import io
import time
import base64
from typing import Dict, List, Tuple
import plotly.express as px
import numpy as np
import requests
import feedparser
from textblob import TextBlob
import json

# Page config
st.set_page_config(
    page_title="Trading System v3.0",
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
    .unrealized-pnl-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .unrealized-profit {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .unrealized-loss {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
</style>
""", unsafe_allow_html=True)


# ==================== GITHUB STORAGE MANAGER ====================

class GitHubStorage:
    """Manage portfolio and stock list storage on GitHub"""
    
    def __init__(self):
        # GitHub configuration from Streamlit secrets or session state
        if 'github_config' not in st.session_state:
            st.session_state.github_config = {
                'token': '',
                'repo': '',  # Format: username/repo
                'portfolio_file': 'trading_portfolio.db',
                'stocklist_file': 'stock_list.csv'
            }
    
    def configure(self, token: str, repo: str, portfolio_file: str = None, stocklist_file: str = None):
        """Configure GitHub settings"""
        st.session_state.github_config['token'] = token
        st.session_state.github_config['repo'] = repo
        if portfolio_file:
            st.session_state.github_config['portfolio_file'] = portfolio_file
        if stocklist_file:
            st.session_state.github_config['stocklist_file'] = stocklist_file
    
    def is_configured(self) -> bool:
        """Check if GitHub is properly configured"""
        config = st.session_state.github_config
        return bool(config['token'] and config['repo'])
    
    def _get_headers(self) -> dict:
        """Get GitHub API headers"""
        return {
            'Authorization': f"token {st.session_state.github_config['token']}",
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def _get_api_url(self, filename: str) -> str:
        """Get GitHub API URL for a file"""
        repo = st.session_state.github_config['repo']
        return f"https://api.github.com/repos/{repo}/contents/{filename}"
    
    def save_portfolio_db(self, db_path: str) -> bool:
        """Save portfolio database to GitHub"""
        if not self.is_configured():
            return False
        
        try:
            # Read database file
            with open(db_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            
            filename = st.session_state.github_config['portfolio_file']
            url = self._get_api_url(filename)
            
            # Check if file exists (to get SHA for update)
            response = requests.get(url, headers=self._get_headers())
            
            data = {
                'message': f'Update portfolio: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'content': content
            }
            
            if response.status_code == 200:
                # File exists, need SHA for update
                data['sha'] = response.json()['sha']
            
            # Create or update file
            response = requests.put(url, headers=self._get_headers(), json=data)
            return response.status_code in [200, 201]
        
        except Exception as e:
            st.error(f"Error saving to GitHub: {str(e)}")
            return False
    
    def load_portfolio_db(self, db_path: str) -> bool:
        """Load portfolio database from GitHub"""
        if not self.is_configured():
            return False
        
        try:
            filename = st.session_state.github_config['portfolio_file']
            url = self._get_api_url(filename)
            
            response = requests.get(url, headers=self._get_headers())
            
            if response.status_code == 200:
                content = base64.b64decode(response.json()['content'])
                with open(db_path, 'wb') as f:
                    f.write(content)
                return True
            return False
        
        except Exception as e:
            st.error(f"Error loading from GitHub: {str(e)}")
            return False
    
    def save_stock_list(self, df: pd.DataFrame) -> bool:
        """Save stock list CSV to GitHub"""
        if not self.is_configured():
            return False
        
        try:
            csv_content = df.to_csv(index=False)
            content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
            
            filename = st.session_state.github_config['stocklist_file']
            url = self._get_api_url(filename)
            
            # Check if file exists
            response = requests.get(url, headers=self._get_headers())
            
            data = {
                'message': f'Update stock list: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'content': content
            }
            
            if response.status_code == 200:
                data['sha'] = response.json()['sha']
            
            response = requests.put(url, headers=self._get_headers(), json=data)
            return response.status_code in [200, 201]
        
        except Exception as e:
            st.error(f"Error saving stock list to GitHub: {str(e)}")
            return False
    
    def load_stock_list(self) -> pd.DataFrame:
        """Load stock list CSV from GitHub"""
        if not self.is_configured():
            return pd.DataFrame()
        
        try:
            filename = st.session_state.github_config['stocklist_file']
            url = self._get_api_url(filename)
            
            response = requests.get(url, headers=self._get_headers())
            
            if response.status_code == 200:
                content = base64.b64decode(response.json()['content']).decode('utf-8')
                return pd.read_csv(io.StringIO(content))
            return pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error loading stock list from GitHub: {str(e)}")
            return pd.DataFrame()


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
                'q': f'{query} AND (stock OR share OR market)',
                'from': from_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', [])[:3]:
                    title = article.get('title', '')
                    if title and symbol.lower() in title.lower():
                        try:
                            pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                            date_str = pub_date.strftime('%Y-%m-%d')
                        except:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        articles.append({
                            'title': title[:100],
                            'source': f"🌐 {article.get('source', {}).get('name', 'NewsAPI')}",
                            'date': date_str,
                            'url': article.get('url', ''),
                            'provider': 'NewsAPI'
                        })
        except:
            pass
        
        return articles
    
    def get_comprehensive_news(self, symbol: str, company_name: str, max_total: int = 8) -> List[Dict]:
        """Get news from all sources"""
        all_articles = []
        
        # Fetch from all sources
        all_articles.extend(self.fetch_newsapi(symbol, company_name))
        all_articles.extend(self.fetch_google_news(symbol, company_name, 3))
        all_articles.extend(self.fetch_economic_times(symbol, company_name, 2))
        all_articles.extend(self.fetch_moneycontrol(symbol, company_name, 2))
        
        # Remove duplicates based on title similarity
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title_lower = article['title'].lower()[:50]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        # Sort by date
        unique_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return unique_articles[:max_total]


# ==================== DATABASE MANAGER ====================

class PortfolioDB:
    def __init__(self, db_name="portfolio.db"):
        self.db_name = db_name
        self.conn = None
        self.github_storage = GitHubStorage()
        self._init_db()
        self._load_from_github()
    
    def _load_from_github(self):
        """Load database from GitHub if configured"""
        if self.github_storage.is_configured():
            loaded = self.github_storage.load_portfolio_db(self.db_name)
            if loaded:
                st.toast("✅ Portfolio loaded from GitHub!")
    
    def _save_to_github(self):
        """Save database to GitHub if configured"""
        if self.github_storage.is_configured():
            self.conn.commit()  # Ensure changes are saved
            saved = self.github_storage.save_portfolio_db(self.db_name)
            if saved:
                st.toast("✅ Portfolio saved to GitHub!")
    
    def _init_db(self):
        """Initialize database tables"""
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Holdings table
        cursor.execute("""
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
        
        # Transactions table
        cursor.execute("""
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
        
        # Realized P&L table
        cursor.execute("""
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
        
        self.conn.commit()
    
    def add_transaction(self, symbol: str, company_name: str, trans_type: str, 
                       quantity: float, price: float, trans_date: date, notes: str = ""):
        """Add a transaction and update holdings"""
        cursor = self.conn.cursor()
        total_amount = quantity * price
        
        # Add transaction
        cursor.execute("""
            INSERT INTO transactions (symbol, company_name, transaction_type, quantity, 
                                    price, total_amount, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, company_name, trans_type, quantity, price, total_amount, trans_date, notes))
        
        # Update holdings
        if trans_type == "BUY":
            cursor.execute("SELECT quantity, avg_price FROM holdings WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result:
                old_qty, old_avg = result
                new_qty = old_qty + quantity
                new_invested = (old_qty * old_avg) + (quantity * price)
                new_avg = new_invested / new_qty
                
                cursor.execute("""
                    UPDATE holdings 
                    SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                """, (new_qty, new_avg, new_invested, symbol))
            else:
                cursor.execute("""
                    INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, company_name, quantity, price, total_amount))
        
        elif trans_type == "SELL":
            cursor.execute("SELECT quantity, avg_price FROM holdings WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result:
                old_qty, avg_price = result
                
                if old_qty >= quantity:
                    # Calculate realized P&L
                    buy_value = quantity * avg_price
                    sell_value = quantity * price
                    pnl = sell_value - buy_value
                    pnl_pct = (pnl / buy_value * 100) if buy_value > 0 else 0
                    
                    # Add to realized P&L
                    cursor.execute("""
                        INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price, 
                                                sell_price, profit_loss, profit_loss_pct, sell_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, company_name, quantity, avg_price, price, pnl, pnl_pct, trans_date))
                    
                    # Update or remove from holdings
                    new_qty = old_qty - quantity
                    if new_qty > 0:
                        new_invested = new_qty * avg_price
                        cursor.execute("""
                            UPDATE holdings 
                            SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE symbol = ?
                        """, (new_qty, new_invested, symbol))
                    else:
                        cursor.execute("DELETE FROM holdings WHERE symbol = ?", (symbol,))
        
        self.conn.commit()
        self._save_to_github()
    
    def get_holdings(self) -> pd.DataFrame:
        """Get all current holdings"""
        return pd.read_sql_query("SELECT * FROM holdings ORDER BY symbol", self.conn)
    
    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        """Get transaction history"""
        return pd.read_sql_query(
            f"SELECT * FROM transactions ORDER BY transaction_date DESC, created_at DESC LIMIT {limit}", 
            self.conn
        )
    
    def get_realized_pnl(self) -> pd.DataFrame:
        """Get realized P&L records"""
        return pd.read_sql_query(
            "SELECT * FROM realized_pnl ORDER BY sell_date DESC", 
            self.conn
        )
    
    def update_holding(self, holding_id: int, quantity: float, avg_price: float):
        """Update a holding"""
        cursor = self.conn.cursor()
        invested_amount = quantity * avg_price
        cursor.execute("""
            UPDATE holdings 
            SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (quantity, avg_price, invested_amount, holding_id))
        self.conn.commit()
        self._save_to_github()
    
    def delete_holding(self, holding_id: int):
        """Delete a holding"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
        self.conn.commit()
        self._save_to_github()
    
    def upload_portfolio_csv(self, df: pd.DataFrame):
        """Upload portfolio from CSV file"""
        cursor = self.conn.cursor()
        
        for _, row in df.iterrows():
            try:
                symbol = str(row['symbol']).strip().upper()
                company_name = str(row.get('company_name', symbol))
                quantity = float(row['quantity'])
                avg_price = float(row['avg_price'])
                invested_amount = quantity * avg_price
                
                # Check if holding exists
                cursor.execute("SELECT id FROM holdings WHERE symbol = ?", (symbol,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing
                    cursor.execute("""
                        UPDATE holdings 
                        SET company_name = ?, quantity = ?, avg_price = ?, 
                            invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    """, (company_name, quantity, avg_price, invested_amount, symbol))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, company_name, quantity, avg_price, invested_amount))
            
            except Exception as e:
                st.warning(f"Error processing {symbol}: {str(e)}")
        
        self.conn.commit()
        self._save_to_github()
    
    def clear_all_holdings(self):
        """Clear all holdings"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM holdings")
        self.conn.commit()
        self._save_to_github()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# ==================== TECHNICAL ANALYSIS ====================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return 50.0


def calculate_macd(prices: pd.Series) -> Tuple[float, float, str]:
    """Calculate MACD"""
    try:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        
        if macd_val > signal_val:
            trend = "BULLISH"
        else:
            trend = "BEARISH"
        
        return macd_val, signal_val, trend
    except:
        return 0.0, 0.0, "NEUTRAL"


def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
    """Calculate moving averages"""
    try:
        return {
            'SMA_20': prices.rolling(window=20).mean().iloc[-1],
            'SMA_50': prices.rolling(window=50).mean().iloc[-1],
            'SMA_200': prices.rolling(window=200).mean().iloc[-1],
            'EMA_20': prices.ewm(span=20, adjust=False).mean().iloc[-1]
        }
    except:
        return {'SMA_20': 0, 'SMA_50': 0, 'SMA_200': 0, 'EMA_20': 0}


def get_stock_info(symbol: str) -> Dict:
    """Get comprehensive stock information"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1y")
        
        if hist.empty:
            ticker = yf.Ticker(f"{symbol}.BO")
            hist = ticker.history(period="1y")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            info = ticker.info
            
            return {
                'valid': True,
                'name': info.get('longName', info.get('shortName', symbol)),
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
            }
    except:
        pass
    
    return {'valid': False, 'name': symbol, 'current_price': 0}


def analyze_stock(symbol: str) -> Dict:
    """Comprehensive stock analysis"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1y")
        
        if hist.empty:
            ticker = yf.Ticker(f"{symbol}.BO")
            hist = ticker.history(period="1y")
        
        if hist.empty:
            return {'error': 'No data available'}
        
        current_price = hist['Close'].iloc[-1]
        rsi = calculate_rsi(hist['Close'])
        macd_val, signal_val, macd_trend = calculate_macd(hist['Close'])
        mas = calculate_moving_averages(hist['Close'])
        
        # Volume analysis
        avg_volume = hist['Volume'].mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_surge = (current_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
        
        # Price momentum
        week_change = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else 0
        month_change = ((current_price - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100) if len(hist) >= 20 else 0
        
        # Signal generation
        signals = []
        if rsi < 30:
            signals.append("🟢 OVERSOLD (RSI)")
        elif rsi > 70:
            signals.append("🔴 OVERBOUGHT (RSI)")
        
        if macd_trend == "BULLISH":
            signals.append("🟢 BULLISH (MACD)")
        else:
            signals.append("🔴 BEARISH (MACD)")
        
        if current_price > mas['SMA_50']:
            signals.append("🟢 ABOVE SMA50")
        else:
            signals.append("🔴 BELOW SMA50")
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'macd': macd_val,
            'signal': signal_val,
            'macd_trend': macd_trend,
            'mas': mas,
            'volume_surge': volume_surge,
            'week_change': week_change,
            'month_change': month_change,
            'signals': signals,
            'data': hist
        }
    
    except Exception as e:
        return {'error': str(e)}


def format_currency(amount: float) -> str:
    """Format currency in Indian style"""
    if amount < 0:
        return f"-₹{abs(amount):,.2f}"
    return f"₹{amount:,.2f}"


# ==================== MAIN APP ====================

def main():
    st.markdown("<h1 class='main-header'>📊 UNIFIED TRADING SYSTEM v3.0</h1>", unsafe_allow_html=True)
    
    # Initialize database
    db = PortfolioDB()
    github = db.github_storage
    news_agg = IndianNewsAggregator()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    
    # GitHub Configuration Section
    with st.sidebar.expander("⚙️ GitHub Configuration", expanded=False):
        st.write("**Save portfolio & stock list on GitHub**")
        st.caption("Your data will be accessible from any device")
        
        github_token = st.text_input(
            "GitHub Token",
            value=st.session_state.github_config.get('token', ''),
            type="password",
            help="Create at: github.com/settings/tokens"
        )
        
        github_repo = st.text_input(
            "Repository (username/repo)",
            value=st.session_state.github_config.get('repo', ''),
            help="Example: johndoe/trading-data"
        )
        
        if st.button("💾 Save GitHub Config"):
            github.configure(github_token, github_repo)
            st.success("✅ GitHub configured!")
            st.rerun()
        
        if github.is_configured():
            st.success("✅ GitHub Connected")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Load Portfolio"):
                    if github.load_portfolio_db(db.db_name):
                        st.success("✅ Loaded from GitHub!")
                        st.rerun()
            with col2:
                if st.button("☁️ Save Portfolio"):
                    if github.save_portfolio_db(db.db_name):
                        st.success("✅ Saved to GitHub!")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Portfolio Manager", "🔍 Stock Scanner", "📈 Stock Analysis",
         "📰 News Feed", "📁 Upload Files", "➕ Add Transaction",
         "📜 Transaction History", "💰 Realized P&L"]
    )
    
    # ==================== PORTFOLIO MANAGER ====================
    if page == "🏠 Portfolio Manager":
        st.header("Portfolio Manager")
        
        # Auto-refresh toggle
        col1, col2 = st.columns([1, 4])
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh")
        with col2:
            if auto_refresh:
                refresh_interval = st.slider("Interval (seconds)", 30, 300, 60, 30)
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
            # ==================== UNREALIZED P&L AT TOP ====================
            total_unrealized_pnl = 0
            total_unrealized_pnl_pct = 0
            total_current_value = 0
            total_invested = holdings['invested_amount'].sum()
            
            # Calculate unrealized P&L
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    pnl = current_value - row['invested_amount']
                    total_unrealized_pnl += pnl
                    total_current_value += current_value
                except:
                    pass
            
            if total_invested > 0:
                total_unrealized_pnl_pct = (total_unrealized_pnl / total_invested * 100)
            
            # Display Unrealized P&L Box
            pnl_class = "unrealized-profit" if total_unrealized_pnl >= 0 else "unrealized-loss"
            
            st.markdown(f"""
                <div class='unrealized-pnl-box {pnl_class}'>
                    <h2 style='margin: 0; font-size: 1.2rem;'>💼 UNREALIZED P&L</h2>
                    <h1 style='margin: 10px 0; font-size: 2.5rem;'>{format_currency(total_unrealized_pnl)}</h1>
                    <h3 style='margin: 0; font-size: 1.5rem;'>({total_unrealized_pnl_pct:+.2f}%)</h3>
                    <hr style='margin: 15px 0; border-color: rgba(255,255,255,0.3);'>
                    <div style='display: flex; justify-content: space-around; margin-top: 10px;'>
                        <div>
                            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>Invested</p>
                            <p style='margin: 0; font-size: 1.2rem; font-weight: bold;'>{format_currency(total_invested)}</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>Current Value</p>
                            <p style='margin: 0; font-size: 1.2rem; font-weight: bold;'>{format_currency(total_current_value)}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Refresh button
            col_refresh, col_clear = st.columns([1, 4])
            with col_refresh:
                if st.button("🔄 Refresh Now"):
                    st.rerun()
            
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
            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        
        else:
            st.info("📤 No holdings yet. Go to 'Upload Files' to import your portfolio!")
    
    # ==================== STOCK SCANNER ====================
    elif page == "🔍 Stock Scanner":
        st.header("Stock Scanner")
        
        # Check if stock list exists on GitHub
        if github.is_configured():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("📂 Stock list will be saved to GitHub for cross-device access")
            with col2:
                if st.button("🔄 Load from GitHub"):
                    loaded_df = github.load_stock_list()
                    if not loaded_df.empty:
                        st.session_state.stock_list_df = loaded_df
                        st.success(f"✅ Loaded {len(loaded_df)} stocks from GitHub!")
                        st.rerun()
        
        # File upload
        uploaded_file = st.file_uploader("📁 Upload Stock List (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.stock_list_df = df
                
                # Save to GitHub if configured
                if github.is_configured():
                    if github.save_stock_list(df):
                        st.success(f"✅ Stock list saved to GitHub! ({len(df)} stocks)")
                    else:
                        st.warning("⚠️ Could not save to GitHub. Check your configuration.")
                
                st.success(f"✅ Loaded {len(df)} stocks!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display and scan
        if 'stock_list_df' in st.session_state and not st.session_state.stock_list_df.empty:
            df = st.session_state.stock_list_df
            
            st.write(f"**Stocks loaded:** {len(df)}")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                scan_type = st.selectbox("Scan Type", ["Quick Overview", "Technical Signals", "Volume Surge"])
            with col2:
                max_stocks = st.slider("Max stocks to scan", 5, min(50, len(df)), 10)
            with col3:
                sort_by = st.selectbox("Sort by", ["Symbol", "RSI", "Volume"])
            
            if st.button("🚀 Start Scan", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.head(max_stocks).iterrows():
                    symbol = row.get('Symbol', row.get('symbol', ''))
                    if symbol:
                        analysis = analyze_stock(symbol)
                        
                        if 'error' not in analysis:
                            results.append({
                                'Symbol': symbol,
                                'Price': analysis['current_price'],
                                'RSI': analysis['rsi'],
                                'MACD': analysis['macd_trend'],
                                'Week%': analysis['week_change'],
                                'Volume': analysis['volume_surge'],
                                'Signal': ' | '.join(analysis['signals'][:2])
                            })
                    
                    progress_bar.progress((idx + 1) / max_stocks)
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Sort
                    if sort_by == "RSI":
                        results_df = results_df.sort_values('RSI')
                    elif sort_by == "Volume":
                        results_df = results_df.sort_values('Volume', ascending=False)
                    
                    st.subheader("📊 Scan Results")
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
        else:
            st.info("📤 Upload a stock list file to start scanning")
    
    # ==================== STOCK ANALYSIS ====================
    elif page == "📈 Stock Analysis":
        st.header("Stock Analysis")
        
        symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE)", "").upper()
        
        if symbol and st.button("🔍 Analyze", type="primary"):
            with st.spinner(f"Analyzing {symbol}..."):
                analysis = analyze_stock(symbol)
                
                if 'error' in analysis:
                    st.error(f"❌ {analysis['error']}")
                else:
                    info = get_stock_info(symbol)
                    
                    # Header
                    st.markdown(f"## {info['name']}")
                    st.markdown(f"**Symbol:** {symbol} | **Sector:** {info.get('sector', 'N/A')}")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
                    with col2:
                        st.metric("RSI", f"{analysis['rsi']:.1f}")
                    with col3:
                        st.metric("Week Change", f"{analysis['week_change']:.2f}%")
                    with col4:
                        st.metric("Volume Surge", f"{analysis['volume_surge']:.1f}%")
                    
                    st.divider()
                    
                    # Signals
                    st.subheader("📊 Trading Signals")
                    for signal in analysis['signals']:
                        st.write(signal)
                    
                    st.divider()
                    
                    # Technical indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 MACD")
                        st.write(f"**Trend:** {analysis['macd_trend']}")
                        st.write(f"**MACD:** {analysis['macd']:.2f}")
                        st.write(f"**Signal:** {analysis['signal']:.2f}")
                    
                    with col2:
                        st.subheader("📉 Moving Averages")
                        mas = analysis['mas']
                        st.write(f"**SMA 20:** ₹{mas['SMA_20']:.2f}")
                        st.write(f"**SMA 50:** ₹{mas['SMA_50']:.2f}")
                        st.write(f"**SMA 200:** ₹{mas['SMA_200']:.2f}")
                    
                    st.divider()
                    
                    # Price chart
                    st.subheader("📊 Price Chart (1 Year)")
                    hist_data = analysis['data']
                    
                    fig = px.line(
                        hist_data, 
                        x=hist_data.index, 
                        y='Close',
                        title=f"{symbol} - Price History"
                    )
                    fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== NEWS FEED ====================
    elif page == "📰 News Feed":
        st.header("News Feed")
        
        symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE)", "").upper()
        
        if symbol and st.button("🔍 Get News", type="primary"):
            with st.spinner(f"Fetching news for {symbol}..."):
                stock_info = get_stock_info(symbol)
                company_name = stock_info['name']
                
                articles = news_agg.get_comprehensive_news(symbol, company_name)
                
                if articles:
                    st.success(f"✅ Found {len(articles)} articles")
                    
                    for article in articles:
                        with st.container():
                            st.markdown(f"### {article['title']}")
                            st.caption(f"{article['source']} | {article['date']}")
                            if article['url']:
                                st.markdown(f"[Read more →]({article['url']})")
                            st.divider()
                else:
                    st.warning("No recent news found")
    
    # ==================== UPLOAD FILES ====================
    elif page == "📁 Upload Files":
        st.header("Upload Portfolio")
        
        upload_method = st.radio(
            "Choose upload method:",
            ["Upload from Computer", "Manual Entry"]
        )
        
        if upload_method == "Upload from Computer":
            st.info("💡 **Supported formats:** CSV or Excel files")
            st.info("📋 **Required columns:** symbol, company_name (optional), quantity, avg_price")
            
            uploaded_file = st.file_uploader(
                "Choose your portfolio file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file with your portfolio data"
            )
            
            if uploaded_file:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.write("**Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Validate columns
                    required_cols = ['symbol', 'quantity', 'avg_price']
                    missing_cols = [col for col in required_cols if col not in [c.lower() for c in df.columns]]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        st.info("Your file must have: symbol, quantity, avg_price")
                    else:
                        # Normalize column names
                        df.columns = df.columns.str.lower()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            replace_existing = st.checkbox("Replace existing portfolio", value=False)
                        with col2:
                            if st.button("📤 Upload Portfolio", type="primary"):
                                if replace_existing:
                                    db.clear_all_holdings()
                                
                                db.upload_portfolio_csv(df)
                                st.success(f"✅ Portfolio uploaded successfully! ({len(df)} holdings)")
                                time.sleep(1)
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        else:  # Manual Entry
            st.info("Enter holdings manually")
            
            with st.form("manual_upload_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    symbol = st.text_input("Symbol (e.g., RELIANCE)").upper()
                    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
                
                with col2:
                    company_name = st.text_input("Company Name (optional)")
                    avg_price = st.number_input("Average Price (₹)", min_value=0.01, value=100.0, step=0.01)
                
                if st.form_submit_button("➕ Add to Portfolio", type="primary"):
                    if symbol:
                        if not company_name:
                            info = get_stock_info(symbol)
                            company_name = info['name'] if info['valid'] else symbol
                        
                        # Create dataframe
                        df = pd.DataFrame([{
                            'symbol': symbol,
                            'company_name': company_name,
                            'quantity': quantity,
                            'avg_price': avg_price
                        }])
                        
                        db.upload_portfolio_csv(df)
                        st.success(f"✅ {symbol} added to portfolio!")
                        time.sleep(1)
                        st.rerun()
        
        st.divider()
        
        # Export portfolio
        st.subheader("📥 Export Portfolio")
        holdings = db.get_holdings()
        
        if not holdings.empty:
            csv = holdings[['symbol', 'company_name', 'quantity', 'avg_price', 'invested_amount']].to_csv(index=False)
            st.download_button(
                "📥 Download Portfolio CSV",
                csv,
                f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("No portfolio data to export")
    
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
                    # Balloon animation removed as requested
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
    
    # Footer
    st.sidebar.divider()
    st.sidebar.success("✨ **v3.0 ENHANCED:**\n- Unrealized P&L ✅\n- GitHub Storage ✅\n- CSV Upload ✅")
    st.sidebar.caption("Stock Scanner - By Ashish Gupta")


if __name__ == "__main__":
    main()
