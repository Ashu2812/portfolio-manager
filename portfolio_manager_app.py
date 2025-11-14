"""
UNIFIED TRADING SYSTEM v2.1 - Enhanced with News Integration
All-in-One Trading Platform with Multi-Source News & Portfolio Management

NEW FEATURES:
- Multi-source News Integration (Google News, Economic Times, BSE, Moneycontrol)
- Tile-based Portfolio View (compact, no expansion needed)
- Edit/Delete Holdings directly from tiles
- Remove incorrect portfolio entries

STRATEGY: 100% INTACT - SMA 9/21 Crossover + Volume (>1.5x 21-day avg)
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
    page_icon="ðŸ“Š",
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
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .portfolio-tile {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .tile-header {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .tile-content {
        font-size: 12px;
        line-height: 1.4;
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


# ==================== NEWS AGGREGATOR ====================

class IndianNewsAggregator:
    """
    Multi-source news aggregator for Indian stocks
    Sources: Google News, Economic Times, BSE, Moneycontrol
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def fetch_google_news(self, symbol: str, company_name: str, max_articles: int = 3) -> List[Dict]:
        """Fetch news from Google News RSS"""
        articles = []
        try:
            # Try company name first, then symbol
            queries = [company_name, symbol]
            
            for query in queries[:1]:  # Try first query only
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
                    
                    # Check relevance
                    text_lower = title.lower()
                    if symbol.lower() in text_lower or any(word in text_lower for word in company_name.lower().split()[:2]):
                        articles.append({
                            'title': title[:100],
                            'source': f"ðŸ“° {source}",
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
                            'source': 'ðŸ“Š Economic Times',
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
                        'source': 'ðŸ’° Moneycontrol',
                        'date': date_str,
                        'url': link,
                        'provider': 'Moneycontrol'
                    })
        except:
            pass
        
        return articles[:max_articles]
    
    def fetch_bse_announcements(self, symbol: str) -> List[Dict]:
        """Fetch BSE corporate announcements"""
        articles = []
        try:
            # For BSE, we need scrip code. This is a simplified version
            # In production, you'd map symbols to BSE scrip codes
            url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
            
            params = {
                'strCat': '-1',
                'strPrevDate': (datetime.now() - timedelta(days=7)).strftime('%Y%m%d'),
                'strScrip': '',  # Would need actual scrip code
                'strSearch': 'S',
                'strToDate': datetime.now().strftime('%Y%m%d'),
                'strType': 'C'
            }
            
            # This is a placeholder - actual implementation would need scrip mapping
            pass
        except:
            pass
        
        return articles
    
    def aggregate_news(self, symbol: str, company_name: str) -> Dict:
        """Aggregate news from all sources with sentiment"""
        all_articles = []
        
        # Fetch from all sources
        all_articles.extend(self.fetch_google_news(symbol, company_name))
        all_articles.extend(self.fetch_economic_times(symbol, company_name))
        all_articles.extend(self.fetch_moneycontrol(symbol, company_name))
        
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
            sentiment_label = 'ðŸŸ¢ Positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'ðŸ”´ Negative'
        else:
            sentiment_label = 'âšª Neutral'
        
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
        
        # Holdings table
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
        
        # Transactions table
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
        
        # Realized P&L table
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
        
        # Stock watchlist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                company_name TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                
                # Check if already exists
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
        
        # Try different column names
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
    Analyze stock for SMA crossover and volume - EXACT ORIGINAL LOGIC
    STRATEGY: NO CHANGES - Uses > operator, checks both days, excludes today from avg
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 30:
            return None
        
        # Calculate SMAs
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
        
        # Volume analysis - ORIGINAL LOGIC: > operator, both days checked
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
        return f"â‚¹{amount/10000000:.2f}Cr"
    elif amount >= 100000:
        return f"â‚¹{amount/100000:.2f}L"
    else:
        return f"â‚¹{amount:,.2f}"


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
    
    # Initialize news aggregator
    if 'news_aggregator' not in st.session_state:
        st.session_state.news_aggregator = IndianNewsAggregator()
    
    db = st.session_state.db
    news_agg = st.session_state.news_aggregator
    
    # Header
    st.markdown('<p class="main-header">ðŸ“Š Unified Trading System v2.1</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Module", [
        "ðŸ  Dashboard",
        "ðŸ“¤ Upload Files",
        "ðŸ” Stock Scanner",
        "ðŸ’¼ Portfolio Manager",
        "âž• Add Transaction",
        "ðŸ“œ Transaction History",
        "ðŸ’° Realized P&L"
    ])
    
    # ==================== DASHBOARD ====================
    if page == "ðŸ  Dashboard":
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
            st.subheader("ðŸ“Š Recent Activity")
            recent_trans = db.get_transactions(limit=5)
            if not recent_trans.empty:
                for _, row in recent_trans.iterrows():
                    trans_type = "ðŸŸ¢ BUY" if row['transaction_type'] == 'BUY' else "ðŸ”´ SELL"
                    st.text(f"{trans_type} {row['symbol']} - {row['quantity']:.0f} @ â‚¹{row['price']:.2f}")
            else:
                st.info("No transactions yet")
        
        with col2:
            st.subheader("â„¹ï¸ System Info")
            st.info("âœ¨ **New Features**\n- News Integration\n- Tile Portfolio View\n- Edit/Delete Holdings")
    
    # ==================== UPLOAD FILES ====================
    elif page == "ðŸ“¤ Upload Files":
        st.header("Upload Excel Files")
        
        with st.expander("ðŸ“‹ Upload Stock List (for Scanner)", expanded=True):
            st.info("Upload Excel with stocks to scan. Expected column: 'Symbol'")
            stock_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], key='stock_list')
            
            if stock_file:
                symbols = load_stock_list_from_excel(stock_file)
                if symbols:
                    st.success(f"âœ… Loaded {len(symbols)} stocks successfully!")
                    st.session_state['stock_list'] = symbols
                    
                    with st.expander("Preview Stocks"):
                        st.write(", ".join(symbols[:50]))
                        if len(symbols) > 50:
                            st.write(f"... and {len(symbols)-50} more")
        
        with st.expander("ðŸ’¼ Upload Portfolio (Bulk Import)", expanded=True):
            st.info("Upload Excel with: Symbol, Company (optional), Quantity, Avg Price")
            portfolio_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], key='portfolio')
            
            if portfolio_file:
                df = load_portfolio_from_excel(portfolio_file)
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("âœ… Import Portfolio", type="primary"):
                        imported, errors = db.bulk_import_portfolio(df)
                        
                        if imported > 0:
                            st.success(f"âœ… Imported {imported} holdings successfully!")
                            st.balloons()
                        
                        if errors:
                            with st.expander(f"âš ï¸ {len(errors)} errors occurred"):
                                for error in errors:
                                    st.warning(error)
    
    # ==================== STOCK SCANNER WITH NEWS ====================
    elif page == "ðŸ” Stock Scanner":
        st.header("Stock Market Scanner with News")
        st.info("ðŸŽ¯ Scans for: SMA 9/21 crossover (last 5 days) + High Volume (>1.5x 21-day avg)")
        
        input_method = st.radio("Select input method:", ["Upload Excel", "Manual Entry", "Use Uploaded List"])
        
        symbols_to_scan = []
        
        if input_method == "Upload Excel":
            file = st.file_uploader("Upload Excel with stocks", type=['xlsx', 'xls'])
            if file:
                symbols_to_scan = load_stock_list_from_excel(file)
                if symbols_to_scan:
                    st.success(f"âœ… Loaded {len(symbols_to_scan)} stocks")
        
        elif input_method == "Use Uploaded List":
            if 'stock_list' in st.session_state and st.session_state['stock_list']:
                symbols_to_scan = st.session_state['stock_list']
                st.success(f"âœ… Using {len(symbols_to_scan)} stocks from uploaded list")
            else:
                st.warning("âš ï¸ No stock list uploaded. Go to 'Upload Files' first!")
        
        else:
            stock_input = st.text_area("Enter stock symbols (one per line)",
                                      "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK",
                                      height=200)
            symbols_to_scan = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
        
        scan_button = st.button("ðŸ” Start Scanning with News", type="primary")
        
        if scan_button and symbols_to_scan:
            st.info(f"â³ Scanning {len(symbols_to_scan)} stocks with news... This may take a few minutes.")
            
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
                        # Fetch news for qualifying stocks
                        news = news_agg.aggregate_news(symbol, result['company_name'])
                        result['news'] = news
                        
                        if result['crossover_type'] == 'BULLISH':
                            bullish_signals.append(result)
                        else:
                            bearish_signals.append(result)
                
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(0.5)  # Slightly slower for news fetching
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"âœ… Scan complete! Found {len(bullish_signals)} bullish and {len(bearish_signals)} bearish signals")
            
            # Display Bullish Signals with News
            if bullish_signals:
                st.subheader("ðŸŸ¢ Bullish Signals")
                for sig in bullish_signals:
                    with st.expander(f"ðŸŸ¢ {sig['symbol']} - â‚¹{sig['current_price']:.2f}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Trend:** {sig['trend']}")
                            st.write(f"**Crossover:** {sig['crossover_day']} days ago")
                        with col2:
                            st.write(f"**SMA9:** â‚¹{sig['sma9']:.2f}")
                            st.write(f"**SMA21:** â‚¹{sig['sma21']:.2f}")
                        
                        st.write(f"**ðŸ“Š Volume Analysis:**")
                        st.write(f"  â€¢ Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'ðŸ”¥' if sig['high_volume_today'] else ''}")
                        st.write(f"  â€¢ Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'ðŸ”¥' if sig['high_volume_yesterday'] else ''}")
                        
                        # News section
                        news = sig.get('news', {})
                        st.write(f"**ðŸ“° News Sentiment:** {news.get('sentiment_label', 'N/A')} (Score: {news.get('sentiment_score', 0):.3f})")
                        
                        articles = news.get('articles', [])
                        if articles:
                            st.write(f"**Latest Headlines ({len(articles)}):**")
                            for idx, article in enumerate(articles[:3], 1):
                                st.write(f"{idx}. [{article['title']}]({article['url']})")
                                st.caption(f"   {article['source']} - {article['date']}")
                        else:
                            st.caption("â„¹ï¸ No recent news found")
                        
                        st.success("ðŸ’¡ **Recommendation:** Consider BUY with stop loss below â‚¹{:.2f} (SMA21)".format(sig['sma21']))
            
            # Display Bearish Signals with News
            if bearish_signals:
                st.subheader("ðŸ”´ Bearish Signals")
                for sig in bearish_signals:
                    with st.expander(f"ðŸ”´ {sig['symbol']} - â‚¹{sig['current_price']:.2f}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Trend:** {sig['trend']}")
                            st.write(f"**Crossover:** {sig['crossover_day']} days ago")
                        with col2:
                            st.write(f"**SMA9:** â‚¹{sig['sma9']:.2f}")
                            st.write(f"**SMA21:** â‚¹{sig['sma21']:.2f}")
                        
                        st.write(f"**ðŸ“Š Volume Analysis:**")
                        st.write(f"  â€¢ Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'ðŸ”¥' if sig['high_volume_today'] else ''}")
                        st.write(f"  â€¢ Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'ðŸ”¥' if sig['high_volume_yesterday'] else ''}")
                        
                        # News section
                        news = sig.get('news', {})
                        st.write(f"**ðŸ“° News Sentiment:** {news.get('sentiment_label', 'N/A')} (Score: {news.get('sentiment_score', 0):.3f})")
                        
                        articles = news.get('articles', [])
                        if articles:
                            st.write(f"**Latest Headlines ({len(articles)}):**")
                            for idx, article in enumerate(articles[:3], 1):
                                st.write(f"{idx}. [{article['title']}]({article['url']})")
                                st.caption(f"   {article['source']} - {article['date']}")
                        else:
                            st.caption("â„¹ï¸ No recent news found")
                        
                        st.error("ðŸ’¡ **Recommendation:** Consider SELL/SHORT with stop loss above â‚¹{:.2f} (SMA21)".format(sig['sma21']))
            
            if not bullish_signals and not bearish_signals:
                st.info("â„¹ï¸ No signals found matching criteria (crossover within 5 days + high volume >1.5x)")
    
    # ==================== PORTFOLIO MANAGER - TILE VIEW ====================
    elif page == "ðŸ’¼ Portfolio Manager":
        st.header("Portfolio Holdings - Tile View")
        
        # Initialize edit/delete mode in session state
        if 'edit_mode' not in st.session_state:
            st.session_state.edit_mode = None
        if 'delete_mode' not in st.session_state:
            st.session_state.delete_mode = None
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
            # Calculate total unrealised P/L first
            total_invested = 0
            total_current_value = 0
            error_symbols = []
            
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    
                    total_invested += row['invested_amount']
                    total_current_value += current_value
                except Exception as e:
                    error_symbols.append(row['symbol'])
                    # Use avg_price as fallback for current price
                    current_value = row['avg_price'] * row['quantity']
                    total_invested += row['invested_amount']
                    total_current_value += current_value
            
            total_unrealised_pnl = total_current_value - total_invested
            total_pnl_pct = (total_unrealised_pnl / total_invested * 100) if total_invested > 0 else 0
            
            # Display Unrealised P/L Summary at the top with styling
            st.markdown("""
            <style>
            .pnl-summary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .pnl-positive {
                color: #4ade80;
                font-weight: bold;
            }
            .pnl-negative {
                color: #f87171;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a prominent P/L display
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                st.metric(
                    label="💰 Total Invested",
                    value=f"₹{total_invested:,.2f}"
                )
            
            with col_summary2:
                st.metric(
                    label="📊 Current Value",
                    value=f"₹{total_current_value:,.2f}"
                )
            
            with col_summary3:
                # Color the metric based on profit/loss
                delta_color = "normal" if total_unrealised_pnl >= 0 else "inverse"
                pnl_emoji = "📈" if total_unrealised_pnl >= 0 else "📉"
                st.metric(
                    label=f"{pnl_emoji} Unrealised P/L",
                    value=f"₹{total_unrealised_pnl:,.2f}",
                    delta=f"{total_pnl_pct:.2f}%",
                    delta_color=delta_color
                )
            
            # Show warning if some symbols had errors
            if error_symbols:
                st.warning(f"⚠️ Could not fetch live prices for: {', '.join(error_symbols)}. Using average price as fallback.")
            
            st.markdown("---")
            
            # If in edit mode, show edit form at top
            if st.session_state.edit_mode is not None:
                edit_id = st.session_state.edit_mode
                edit_row = holdings[holdings['id'] == edit_id].iloc[0]
                
                st.subheader(f"âœï¸ Editing: {edit_row['symbol']}")
                
                with st.form("edit_form", clear_on_submit=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_qty = st.number_input("Quantity", value=float(edit_row['quantity']), min_value=0.1, step=1.0, key="edit_qty")
                    with col2:
                        new_avg = st.number_input("Avg Price (â‚¹)", value=float(edit_row['avg_price']), min_value=0.01, step=0.01, key="edit_avg")
                    
                    col_save, col_cancel = st.columns([1, 1])
                    with col_save:
                        save_btn = st.form_submit_button("ðŸ’¾ Save Changes", type="primary", use_container_width=True)
                    with col_cancel:
                        cancel_btn = st.form_submit_button("âŒ Cancel", use_container_width=True)
                    
                    if save_btn:
                        db.update_holding(edit_id, new_qty, new_avg)
                        st.session_state.edit_mode = None
                        st.success("âœ… Holdings updated successfully!")
                        st.rerun()
                    
                    if cancel_btn:
                        st.session_state.edit_mode = None
                        st.rerun()
                
                st.divider()
            
            # If in delete mode, show confirmation at top
            if st.session_state.delete_mode is not None:
                delete_id = st.session_state.delete_mode
                delete_row = holdings[holdings['id'] == delete_id].iloc[0]
                
                st.error(f"âš ï¸ Delete {delete_row['symbol']} - {delete_row['company_name']}?")
                st.warning(f"This will permanently remove {delete_row['quantity']:.0f} shares worth â‚¹{delete_row['invested_amount']:,.2f}")
                
                col_yes, col_no = st.columns([1, 1])
                with col_yes:
                    if st.button("âœ… Yes, Delete", type="primary", use_container_width=True, key="confirm_delete"):
                        db.delete_holding(delete_id)
                        st.session_state.delete_mode = None
                        st.success("âœ… Holding deleted successfully!")
                        st.rerun()
                with col_no:
                    if st.button("âŒ No, Keep It", use_container_width=True, key="cancel_delete"):
                        st.session_state.delete_mode = None
                        st.rerun()
                
                st.divider()
            
            # Create tiles in rows of 3
            for idx in range(0, len(holdings), 3):
                cols = st.columns(3)
                
                for col_idx, col in enumerate(cols):
                    if idx + col_idx < len(holdings):
                        row = holdings.iloc[idx + col_idx]
                        
                        with col:
                            # Fetch current price
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
                                    
                                    # Metrics in compact format
                                    st.write(f"**Qty:** {row['quantity']:.0f} | **Avg:** â‚¹{row['avg_price']:.2f}")
                                    st.write(f"**CMP:** â‚¹{current_price:.2f}")
                                    
                                    # P&L with color
                                    pnl_class = "profit" if pnl >= 0 else "loss"
                                    st.markdown(f"<p class='{pnl_class}'>P&L: {format_currency(pnl)} ({pnl_pct:+.2f}%)</p>", unsafe_allow_html=True)
                                    
                                    # Action buttons
                                    col_btn1, col_btn2 = st.columns(2)
                                    with col_btn1:
                                        if st.button("âœï¸ Edit", key=f"edit_btn_{row['id']}", use_container_width=True):
                                            st.session_state.edit_mode = row['id']
                                            st.session_state.delete_mode = None
                                            st.rerun()
                                    with col_btn2:
                                        if st.button("ðŸ—‘ï¸ Delete", key=f"del_btn_{row['id']}", use_container_width=True):
                                            st.session_state.delete_mode = row['id']
                                            st.session_state.edit_mode = None
                                            st.rerun()
                                    
                                    st.divider()
                            
                            except Exception as e:
                                st.error(f"Error loading {row['symbol']}: {str(e)}")
        else:
            st.info("ðŸ“¤ No holdings yet. Go to 'Upload Files' to import your portfolio!")
    
    # ==================== ADD TRANSACTION ====================
    elif page == "âž• Add Transaction":
        st.header("Add Transaction")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                trans_type = st.selectbox("Type", ["BUY", "SELL"])
                symbol = st.text_input("Symbol (e.g., RELIANCE)", "").upper()
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            
            with col2:
                trans_date = st.date_input("Date", value=date.today())
                price = st.number_input("Price (â‚¹)", min_value=0.01, value=100.0, step=0.01)
                notes = st.text_area("Notes (optional)", "")
            
            if symbol:
                stock_info = get_stock_info(symbol)
                if stock_info['valid']:
                    st.success(f"âœ… {stock_info['name']} - Current: â‚¹{stock_info['current_price']:.2f}")
                    company_name = stock_info['name']
                else:
                    st.warning("âš ï¸ Could not fetch stock info. Using symbol as name.")
                    company_name = symbol
            else:
                company_name = ""
            
            total = quantity * price
            st.info(f"ðŸ’° Total Amount: {format_currency(total)}")
            
            submitted = st.form_submit_button("âœ… Add Transaction", type="primary")
            
            if submitted and symbol and company_name:
                try:
                    db.add_transaction(symbol, company_name, trans_type, quantity, price, trans_date, notes)
                    st.success("âœ… Transaction added successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"âŒ Error: {str(e)}")
    
    # ==================== TRANSACTION HISTORY ====================
    elif page == "ðŸ“œ Transaction History":
        st.header("Transaction History")
        
        transactions = db.get_transactions(limit=100)
        
        if not transactions.empty:
            st.dataframe(
                transactions[['transaction_date', 'symbol', 'transaction_type',
                             'quantity', 'price', 'total_amount', 'notes']],
                use_container_width=True,
                height=400
            )
            
            st.info(f"ðŸ“Š Showing last {len(transactions)} transactions")
        else:
            st.info("ðŸ“‹ No transactions yet.")
    
    # ==================== REALIZED P&L ====================
    elif page == "ðŸ’° Realized P&L":
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
            st.info("ðŸ’° No realized P&L yet. Sell some holdings to see booked profits/losses.")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("âœ¨ **New in v2.1:**\n- News Integration\n- Tile Portfolio\n- Edit/Delete")
    st.sidebar.caption("Unified Trading System v2.1")


if __name__ == "__main__":
    main()
