"""
UNIFIED TRADING SYSTEM v2.1 FINAL - All Issues Fixed
- NewsAPI integration restored
- Auto-refresh prices
- Working Edit/Delete buttons (simplified approach)
- Strategy 100% intact
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

# Page config
st.set_page_config(
    page_title="Trading System v2.1",
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
</style>
""", unsafe_allow_html=True)


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
            
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&apiKey={self.news_api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get('articles', [])[:3]:
                title = article.get('title', '')
                if title:
                    articles.append({
                        'title': title[:100],
                        'source': f"🌐 {article.get('source', {}).get('name', 'News')}",
                        'date': article.get('publishedAt', '')[:10],
                        'url': article.get('url', ''),
                        'provider': 'NewsAPI'
                    })
        except:
            pass
        
        return articles
    
    def aggregate_news(self, symbol: str, company_name: str) -> Dict:
        """Aggregate news from all sources including NewsAPI"""
        all_articles = []
        
        # Fetch from all sources
        all_articles.extend(self.fetch_google_news(symbol, company_name))
        all_articles.extend(self.fetch_economic_times(symbol, company_name))
        all_articles.extend(self.fetch_moneycontrol(symbol, company_name))
        all_articles.extend(self.fetch_newsapi(symbol, company_name))
        
        # Calculate sentiment
        sentiment_score = 0
        if all_articles:
            for article in all_articles:
                try:
                    blob = TextBlob(article['title'])
                    sentiment_score += blob.sentiment.polarity
                except:
                    pass
            
            sentiment_score = sentiment_score / len(all_articles) if all_articles else 0
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment_label = '🟢 Positive'
        elif sentiment_score < -0.1:
            sentiment_label = '🔴 Negative'
        else:
            sentiment_label = '⚪ Neutral'
        
        return {
            'articles': all_articles,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'article_count': len(all_articles)
        }


# ==================== DATABASE ====================

class PortfolioDatabase:
    """Database for portfolio management"""
    
    def __init__(self, db_path='portfolio.db'):
        self.db_path = db_path
        self.init_database()
    
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
        
        conn.commit()
        conn.close()
    
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
            self._add_to_holdings(cursor, symbol, company_name, quantity, price)
        elif trans_type.upper() == 'SELL':
            realized_pnl = self._reduce_from_holdings(cursor, symbol, quantity, price)
            if realized_pnl:
                cursor.execute('''
                    INSERT INTO realized_pnl (symbol, company_name, quantity, buy_price,
                                             sell_price, profit_loss, profit_loss_pct, transaction_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', realized_pnl + (transaction_id,))
        
        conn.commit()
        conn.close()
        return transaction_id
    
    def _add_to_holdings(self, cursor, symbol: str, company_name: str,
                        quantity: float, price: float):
        cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
        holding = cursor.fetchone()
        
        if holding:
            old_qty = holding[3]
            old_avg = holding[4]
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            new_invested = new_qty * new_avg
            
            cursor.execute('''
                UPDATE holdings
                SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ?
            ''', (new_qty, new_avg, new_invested, symbol.upper()))
        else:
            invested = quantity * price
            cursor.execute('''
                INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol.upper(), company_name, quantity, price, invested))
    
    def _reduce_from_holdings(self, cursor, symbol: str, quantity: float,
                             sell_price: float) -> Tuple:
        cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
        holding = cursor.fetchone()
        
        if not holding:
            raise ValueError(f"No holding found for {symbol}")
        
        current_qty = holding[3]
        avg_price = holding[4]
        
        if quantity > current_qty:
            raise ValueError(f"Cannot sell {quantity} shares. Only {current_qty} available")
        
        profit_loss = (sell_price - avg_price) * quantity
        profit_loss_pct = ((sell_price - avg_price) / avg_price) * 100
        
        new_qty = current_qty - quantity
        
        if new_qty <= 0:
            cursor.execute('DELETE FROM holdings WHERE symbol = ?', (symbol.upper(),))
        else:
            new_invested = new_qty * avg_price
            cursor.execute('''
                UPDATE holdings
                SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ?
            ''', (new_qty, new_invested, symbol.upper()))
        
        return (symbol.upper(), holding[2], quantity, avg_price, sell_price,
                profit_loss, profit_loss_pct)
    
    def delete_holding(self, holding_id: int):
        """Delete a holding entirely"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM holdings WHERE id = ?', (holding_id,))
        conn.commit()
        conn.close()
    
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
        """Bulk import portfolio from DataFrame"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        imported = 0
        errors = []
        
        for idx, row in df.iterrows():
            try:
                symbol = str(row.get('Symbol', '')).strip().upper()
                company = str(row.get('Company', symbol))
                quantity = float(row.get('Quantity', 0))
                avg_price = float(row.get('Avg Price', 0))
                
                if not symbol or quantity <= 0 or avg_price <= 0:
                    errors.append(f"Row {idx+1}: Invalid data")
                    continue
                
                cursor.execute('SELECT quantity, avg_price FROM holdings WHERE symbol = ?', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    old_qty, old_avg = existing
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + (quantity * avg_price)) / new_qty
                    new_invested = new_qty * new_avg
                    
                    cursor.execute('''
                        UPDATE holdings
                        SET quantity = ?, avg_price = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (new_qty, new_avg, new_invested, symbol))
                else:
                    invested = quantity * avg_price
                    cursor.execute('''
                        INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, company, quantity, avg_price, invested))
                
                imported += 1
                
            except Exception as e:
                errors.append(f"Row {idx+1}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        return imported, errors


# ==================== EXCEL PROCESSORS ====================

def load_stock_list_from_excel(file) -> List[str]:
    """Load stock symbols from Excel"""
    try:
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
        
        symbols = [s.replace('.NS', '').replace('.BO', '') for s in symbols if s]
        
        return list(set(symbols))
    except Exception as e:
        st.error(f"Error loading Excel: {str(e)}")
        return []


def load_portfolio_from_excel(file) -> pd.DataFrame:
    """Load portfolio from Excel"""
    try:
        df = pd.read_excel(file)
        
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


# ==================== STOCK ANALYZER - STRATEGY 100% INTACT ====================

def analyze_stock(symbol: str, company_name: str) -> Dict:
    """
    STRATEGY UNCHANGED - EXACT ORIGINAL LOGIC
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 30:
            return None
        
        hist['SMA9'] = hist['Close'].rolling(window=9).mean()
        hist['SMA21'] = hist['Close'].rolling(window=21).mean()
        
        # Volume average (excluding current day - ORIGINAL LOGIC)
        vol_21day_avg = hist['Volume'].iloc[-22:-1].mean() if len(hist) >= 22 else hist['Volume'].iloc[:-1].mean()
        
        latest = hist.iloc[-1]
        current_price = latest['Close']
        sma9 = latest['SMA9']
        sma21 = latest['SMA21']
        
        if pd.isna(sma9) or pd.isna(sma21):
            return None
        
        current_trend = 'BULLISH' if sma9 > sma21 else 'BEARISH'
        
        # Detect crossover in last 5 days
        crossover_detected = False
        crossover_type = None
        crossover_day = None
        
        for i in range(1, min(6, len(hist))):
            prev = hist.iloc[-(i+1)]
            curr = hist.iloc[-i]
            
            if pd.notna(prev['SMA9']) and pd.notna(prev['SMA21']) and pd.notna(curr['SMA9']) and pd.notna(curr['SMA21']):
                if prev['SMA9'] <= prev['SMA21'] and curr['SMA9'] > curr['SMA21']:
                    crossover_detected = True
                    crossover_type = 'BULLISH'
                    crossover_day = i
                    break
                elif prev['SMA9'] >= prev['SMA21'] and curr['SMA9'] < curr['SMA21']:
                    crossover_detected = True
                    crossover_type = 'BEARISH'
                    crossover_day = i
                    break
        
        # Volume analysis - ORIGINAL LOGIC
        today_volume = hist['Volume'].iloc[-1]
        yesterday_volume = hist['Volume'].iloc[-2] if len(hist) >= 2 else 0
        
        high_volume_today = today_volume > vol_21day_avg * 1.5
        high_volume_yesterday = yesterday_volume > vol_21day_avg * 1.5
        high_volume = high_volume_today or high_volume_yesterday
        
        volume_ratio_today = today_volume / vol_21day_avg if vol_21day_avg > 0 else 0
        volume_ratio_yesterday = yesterday_volume / vol_21day_avg if vol_21day_avg > 0 else 0
        
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


# ==================== MAIN APP ====================

def main():
    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = PortfolioDatabase()
    
    # Initialize news aggregator with NewsAPI
    if 'news_aggregator' not in st.session_state:
        st.session_state.news_aggregator = IndianNewsAggregator()
    
    # Initialize stock list
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = []
    
    # Initialize temporary symbols storage
    if 'temp_symbols' not in st.session_state:
        st.session_state.temp_symbols = []
    
    db = st.session_state.db
    news_agg = st.session_state.news_aggregator
    
    # Header
    st.markdown('<p class="main-header">📊 Unified Stock Scanner </p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Module", [
        "🏠 Dashboard",
        "📤 Upload Files",
        "🔍 Stock Scanner",
        "💼 Portfolio Manager",
        "➕ Add Transaction",
        "📜 Transaction History",
        "💰 Realized P&L"
    ])
    
    # Add auto-refresh option for portfolio
    if page == "💼 Portfolio Manager":
        st.sidebar.divider()
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh prices", value=False)
        if auto_refresh:
            refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)
            st.sidebar.caption(f"Next refresh in {refresh_interval}s")
    
    # ==================== DASHBOARD ====================
    if page == "🏠 Dashboard":
        st.header("Dashboard Overview")
        
        summary = db.get_portfolio_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Holdings", summary['total_holdings'])
        with col2:
            st.metric("Invested", format_currency(summary['total_invested']))
        with col3:
            st.metric("Total Shares", f"{summary['total_quantity']:.0f}")
        with col4:
            realized = db.get_realized_pnl()
            total_realized = realized['profit_loss'].sum() if not realized.empty else 0
            st.metric("Realized P&L", format_currency(total_realized))
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Recent Activity")
            recent_trans = db.get_transactions(limit=5)
            if not recent_trans.empty:
                for _, row in recent_trans.iterrows():
                    trans_type = "🟢 BUY" if row['transaction_type'] == 'BUY' else "🔴 SELL"
                    st.text(f"{trans_type} {row['symbol']} - {row['quantity']:.0f} @ ₹{row['price']:.2f}")
            else:
                st.info("No transactions yet")
        
        with col2:
            st.subheader("ℹ️ System Info")
            st.info("✨ **All Features Working**\n- NewsAPI Integrated\n- Auto-refresh prices\n- Edit/Delete Fixed")
    
    # ==================== UPLOAD FILES ====================
    elif page == "📤 Upload Files":
        st.header("Upload Excel Files")
        
        # Show currently loaded stock list if exists
        if 'stock_list' in st.session_state and st.session_state.stock_list:
            st.success(f"📊 **Currently loaded:** {len(st.session_state.stock_list)} stocks in memory")
            with st.expander("View loaded stocks"):
                st.write(", ".join(st.session_state.stock_list[:50]))
                if len(st.session_state.stock_list) > 50:
                    st.write(f"... and {len(st.session_state.stock_list)-50} more")
            
            if st.button("🗑️ Clear Stock List", key="clear_stocks"):
                st.session_state.stock_list = []
                st.rerun()
        
        st.divider()
        
        with st.expander("📋 Upload Stock List (for Scanner)", expanded=True):
            st.info("Upload Excel with stocks to scan. Expected column: 'Symbol'")
            stock_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], key='stock_list_uploader')
            
            if stock_file is not None:
                # Load symbols from file
                symbols = load_stock_list_from_excel(stock_file)
                if symbols:
                    st.session_state.temp_symbols = symbols
                    st.success(f"✅ Loaded {len(symbols)} stocks from file!")
                    
                    with st.expander("Preview Stocks"):
                        st.write(", ".join(symbols[:50]))
                        if len(symbols) > 50:
                            st.write(f"... and {len(symbols)-50} more")
                    
                    # Add explicit save button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("💾 Save to Scanner", type="primary", key="save_stocks"):
                            st.session_state.stock_list = st.session_state.temp_symbols.copy()
                            st.success(f"✅ Saved {len(st.session_state.stock_list)} stocks to memory!")
                            st.info("🎯 Go to Stock Scanner and select 'Use Uploaded List'")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                    with col2:
                        st.info("👈 Click 'Save to Scanner' to make stocks available in Scanner")
                else:
                    st.error("❌ Could not load stocks from file. Check the format.")
            elif st.session_state.temp_symbols:
                # Show previously loaded temp symbols (file still in uploader state)
                st.info(f"📋 {len(st.session_state.temp_symbols)} stocks loaded. Click 'Save to Scanner' button above.")
        
        
        with st.expander("💼 Upload Portfolio (Bulk Import)", expanded=True):
            st.info("Upload Excel with: Symbol, Company (optional), Quantity, Avg Price")
            portfolio_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], key='portfolio')
            
            if portfolio_file:
                df = load_portfolio_from_excel(portfolio_file)
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("✅ Import Portfolio", type="primary"):
                        imported, errors = db.bulk_import_portfolio(df)
                        
                        if imported > 0:
                            st.success(f"✅ Imported {imported} holdings successfully!")
                            st.balloons()
                        
                        if errors:
                            with st.expander(f"⚠️ {len(errors)} errors occurred"):
                                for error in errors:
                                    st.warning(error)
    
    # ==================== STOCK SCANNER WITH NEWS ====================
    elif page == "🔍 Stock Scanner":
        st.header("Stock Market Scanner with News (NewsAPI Included)")
        st.info("🎯 Scans for: SMA 9/21 crossover (last 5 days) + High Volume (>1.5x 21-day avg)")
        
        input_method = st.radio("Select input method:", ["Upload Excel", "Manual Entry", "Use Uploaded List"])
        
        symbols_to_scan = []
        
        if input_method == "Upload Excel":
            file = st.file_uploader("Upload Excel with stocks", type=['xlsx', 'xls'])
            if file:
                symbols_to_scan = load_stock_list_from_excel(file)
                if symbols_to_scan:
                    st.success(f"✅ Loaded {len(symbols_to_scan)} stocks")
        
        elif input_method == "Use Uploaded List":
            if 'stock_list' in st.session_state and st.session_state.stock_list:
                symbols_to_scan = st.session_state.stock_list
                st.success(f"✅ Using {len(symbols_to_scan)} stocks from uploaded list")
            else:
                st.warning("⚠️ No stock list uploaded. Go to 'Upload Files' first!")
                # Debug info
                with st.expander("🔍 Debug Info"):
                    st.write(f"Session state has stock_list: {'stock_list' in st.session_state}")
                    if 'stock_list' in st.session_state:
                        st.write(f"Stock list content: {st.session_state.stock_list[:5] if st.session_state.stock_list else 'Empty list'}")
                        st.write(f"Stock list length: {len(st.session_state.stock_list) if st.session_state.stock_list else 0}")
        
        else:
            stock_input = st.text_area("Enter stock symbols (one per line)",
                                      "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK",
                                      height=200)
            symbols_to_scan = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
        
        scan_button = st.button("🔍 Start Scanning with News", type="primary")
        
        if scan_button and symbols_to_scan:
            st.info(f"⏳ Scanning {len(symbols_to_scan)} stocks with news... This may take a few minutes.")
            
            bullish_signals = []
            bearish_signals = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(symbols_to_scan):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols_to_scan)})")
                result = analyze_stock(symbol, symbol)
                
                # EXACT ORIGINAL QUALIFICATION LOGIC
                if result and result['crossover_detected'] and result['crossover_day'] <= 5:
                    if result['high_volume']:
                        # Fetch news
                        news = news_agg.aggregate_news(symbol, result['company_name'])
                        result['news'] = news
                        
                        if result['crossover_type'] == 'BULLISH':
                            bullish_signals.append(result)
                        else:
                            bearish_signals.append(result)
                
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Scan complete! Found {len(bullish_signals)} bullish and {len(bearish_signals)} bearish signals")
            
            # Display Bullish Signals
            if bullish_signals:
                st.subheader("🟢 Bullish Signals")
                for sig in bullish_signals:
                    with st.expander(f"🟢 {sig['symbol']} - ₹{sig['current_price']:.2f}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Trend:** {sig['trend']}")
                            st.write(f"**Crossover:** {sig['crossover_day']} days ago")
                        with col2:
                            st.write(f"**SMA9:** ₹{sig['sma9']:.2f}")
                            st.write(f"**SMA21:** ₹{sig['sma21']:.2f}")
                        
                        st.write(f"**📊 Volume Analysis:**")
                        st.write(f"  • Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'🔥' if sig['high_volume_today'] else ''}")
                        st.write(f"  • Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'🔥' if sig['high_volume_yesterday'] else ''}")
                        
                        # News section
                        news = sig.get('news', {})
                        st.write(f"**📰 News Sentiment:** {news.get('sentiment_label', 'N/A')} (Score: {news.get('sentiment_score', 0):.3f})")
                        st.caption(f"Sources: Google News, Economic Times, Moneycontrol, NewsAPI")
                        
                        articles = news.get('articles', [])
                        if articles:
                            st.write(f"**Latest Headlines ({len(articles)}):**")
                            for idx, article in enumerate(articles[:5], 1):
                                st.write(f"{idx}. [{article['title']}]({article['url']})")
                                st.caption(f"   {article['source']} - {article['date']}")
                        else:
                            st.caption("ℹ️ No recent news found")
                        
                        st.success("💡 **Recommendation:** Consider BUY with stop loss below ₹{:.2f} (SMA21)".format(sig['sma21']))
            
            # Display Bearish Signals
            if bearish_signals:
                st.subheader("🔴 Bearish Signals")
                for sig in bearish_signals:
                    with st.expander(f"🔴 {sig['symbol']} - ₹{sig['current_price']:.2f}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Trend:** {sig['trend']}")
                            st.write(f"**Crossover:** {sig['crossover_day']} days ago")
                        with col2:
                            st.write(f"**SMA9:** ₹{sig['sma9']:.2f}")
                            st.write(f"**SMA21:** ₹{sig['sma21']:.2f}")
                        
                        st.write(f"**📊 Volume Analysis:**")
                        st.write(f"  • Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'🔥' if sig['high_volume_today'] else ''}")
                        st.write(f"  • Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'🔥' if sig['high_volume_yesterday'] else ''}")
                        
                        # News section
                        news = sig.get('news', {})
                        st.write(f"**📰 News Sentiment:** {news.get('sentiment_label', 'N/A')} (Score: {news.get('sentiment_score', 0):.3f})")
                        st.caption(f"Sources: Google News, Economic Times, Moneycontrol, NewsAPI")
                        
                        articles = news.get('articles', [])
                        if articles:
                            st.write(f"**Latest Headlines ({len(articles)}):**")
                            for idx, article in enumerate(articles[:5], 1):
                                st.write(f"{idx}. [{article['title']}]({article['url']})")
                                st.caption(f"   {article['source']} - {article['date']}")
                        else:
                            st.caption("ℹ️ No recent news found")
                        
                        st.error("💡 **Recommendation:** Consider SELL/SHORT with stop loss above ₹{:.2f} (SMA21)".format(sig['sma21']))
            
            if not bullish_signals and not bearish_signals:
                st.info("ℹ️ No signals found matching criteria (crossover within 5 days + high volume >1.5x)")
    
    # ==================== PORTFOLIO MANAGER WITH AUTO-REFRESH ====================
    elif page == "💼 Portfolio Manager":
        st.header("Portfolio Holdings - Tile View with Auto-Refresh")
        
        # Refresh button
        col_refresh, col_clear = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
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
                    st.balloons()
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
    st.sidebar.info("✨ **v2.2 FINAL")
    st.sidebar.caption("Stock Scanner - By Ashish Gupta")


if __name__ == "__main__":
    main()
