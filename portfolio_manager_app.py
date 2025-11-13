"""
UNIFIED TRADING SYSTEM v2.0 - Enhanced with Excel Upload
All-in-One Trading Platform with File Upload Support

Features:
- Excel Upload for Stock List (223 stocks)
- Excel Upload for Portfolio (bulk import)
- Stock Market Scanner (SMA + Volume)
- Portfolio Manager (track holdings)
- Transaction Management (buy/sell with P&L)
- Mobile-friendly UI
- Cloud-based with persistent storage

Deploy to Streamlit Cloud → Use forever from anywhere
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

# Page config
st.set_page_config(
    page_title="Trading System v2.0",
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
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


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
    
    def add_to_watchlist(self, symbol: str, company_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO watchlist (symbol, company_name)
                VALUES (?, ?)
            ''', (symbol.upper(), company_name))
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False
        finally:
            conn.close()
        return success
    
    def get_watchlist(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM watchlist ORDER BY added_at DESC', conn)
        conn.close()
        return df
    
    def remove_from_watchlist(self, symbol: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM watchlist WHERE symbol = ?', (symbol.upper(),))
        conn.commit()
        conn.close()
    
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
            # If no header, use first column
            symbols = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        else:
            symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
        
        # Clean symbols
        symbols = [s.replace('.NS', '').replace('.BO', '') for s in symbols if s]
        
        return list(set(symbols))  # Remove duplicates
    except Exception as e:
        st.error(f"Error loading Excel: {str(e)}")
        return []


def load_portfolio_from_excel(file) -> pd.DataFrame:
    """Load portfolio from Excel"""
    try:
        df = pd.read_excel(file)
        
        # Expected columns: Symbol, Company (optional), Quantity, Avg Price
        required_cols = []
        
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
            elif 'price' in col_lower or 'avg' in col_lower or 'cost' in col_lower:
                col_map[col] = 'Avg Price'
        
        df.rename(columns=col_map, inplace=True)
        
        # Validate required columns
        if 'Symbol' not in df.columns or 'Quantity' not in df.columns or 'Avg Price' not in df.columns:
            st.error("Required columns: Symbol, Quantity, Avg Price")
            return pd.DataFrame()
        
        # Add Company if missing
        if 'Company' not in df.columns:
            df['Company'] = df['Symbol']
        
        return df[['Symbol', 'Company', 'Quantity', 'Avg Price']]
        
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return pd.DataFrame()


# ==================== STOCK ANALYZER ====================

def analyze_stock(symbol: str, company_name: str) -> Dict:
    """Analyze stock for SMA crossover and volume - EXACT ORIGINAL LOGIC"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 30:
            return None
        
        # Calculate SMAs
        hist['SMA9'] = hist['Close'].rolling(window=9).mean()
        hist['SMA21'] = hist['Close'].rolling(window=21).mean()
        
        # Calculate volume averages (excluding current day for average calculation)
        # Using iloc[-22:-1] to match original logic exactly
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
                # Bullish crossover: SMA9 crosses above SMA21
                if prev['SMA9'] <= prev['SMA21'] and curr['SMA9'] > curr['SMA21']:
                    crossover_detected = True
                    crossover_type = 'BULLISH'
                    crossover_day = i
                    break
                # Bearish crossover: SMA9 crosses below SMA21
                elif prev['SMA9'] >= prev['SMA21'] and curr['SMA9'] < curr['SMA21']:
                    crossover_detected = True
                    crossover_type = 'BEARISH'
                    crossover_day = i
                    break
        
        # Volume analysis - EXACT ORIGINAL LOGIC
        today_volume = hist['Volume'].iloc[-1]
        yesterday_volume = hist['Volume'].iloc[-2] if len(hist) >= 2 else 0
        
        # CRITICAL: Use > (strictly greater than) not >=
        # CRITICAL: Check BOTH today AND yesterday
        high_volume_today = today_volume > vol_21day_avg * 1.5
        high_volume_yesterday = yesterday_volume > vol_21day_avg * 1.5
        
        # Qualifies if EITHER day has high volume
        high_volume = high_volume_today or high_volume_yesterday
        
        # Calculate ratios for display
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
    
    db = st.session_state.db
    
    # Header
    st.markdown('<p class="main-header">📊 Unified Trading System v2.0</p>', unsafe_allow_html=True)
    
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
        
        # Quick stats
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
            st.subheader("⭐ Watchlist")
            watchlist = db.get_watchlist()
            if not watchlist.empty:
                for _, row in watchlist.iterrows():
                    st.text(f"• {row['symbol']} - {row['company_name']}")
            else:
                st.info("Watchlist empty")
    
    # ==================== UPLOAD FILES ====================
    elif page == "📤 Upload Files":
        st.header("Upload Excel Files")
        
        # Stock List Upload
        with st.expander("📋 Upload Stock List (for Scanner)", expanded=True):
            st.info("Upload Excel with stocks to scan. Expected column: 'Symbol'")
            stock_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'], key='stock_list')
            
            if stock_file:
                symbols = load_stock_list_from_excel(stock_file)
                if symbols:
                    st.success(f"✅ Loaded {len(symbols)} stocks successfully!")
                    st.session_state['stock_list'] = symbols
                    
                    with st.expander("Preview Stocks"):
                        st.write(", ".join(symbols[:50]))
                        if len(symbols) > 50:
                            st.write(f"... and {len(symbols)-50} more")
        
        # Portfolio Upload
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
    
    # ==================== STOCK SCANNER ====================
    elif page == "🔍 Stock Scanner":
        st.header("Stock Market Scanner")
        st.info("🎯 Scans for: SMA 9/21 crossover (last 5 days) + High Volume (>1.5x 21-day avg)")
        
        # Stock input options
        input_method = st.radio("Select input method:", ["Upload Excel", "Manual Entry", "Use Uploaded List"])
        
        symbols_to_scan = []
        
        if input_method == "Upload Excel":
            file = st.file_uploader("Upload Excel with stocks", type=['xlsx', 'xls'])
            if file:
                symbols_to_scan = load_stock_list_from_excel(file)
                if symbols_to_scan:
                    st.success(f"✅ Loaded {len(symbols_to_scan)} stocks")
        
        elif input_method == "Use Uploaded List":
            if 'stock_list' in st.session_state and st.session_state['stock_list']:
                symbols_to_scan = st.session_state['stock_list']
                st.success(f"✅ Using {len(symbols_to_scan)} stocks from uploaded list")
            else:
                st.warning("⚠️ No stock list uploaded. Go to 'Upload Files' first!")
        
        else:  # Manual Entry
            stock_input = st.text_area("Enter stock symbols (one per line)",
                                      "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK",
                                      height=200)
            symbols_to_scan = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
        
        scan_button = st.button("🔍 Start Scanning", type="primary")
        
        if scan_button and symbols_to_scan:
            st.info(f"⏳ Scanning {len(symbols_to_scan)} stocks... This may take a few minutes.")
            
            bullish_signals = []
            bearish_signals = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(symbols_to_scan):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols_to_scan)})")
                result = analyze_stock(symbol, symbol)
                
                # EXACT ORIGINAL QUALIFICATION LOGIC
                if result and result['crossover_detected'] and result['crossover_day'] <= 5:
                    # Must have high volume (today OR yesterday)
                    if result['high_volume']:
                        if result['crossover_type'] == 'BULLISH':
                            bullish_signals.append(result)
                        else:
                            bearish_signals.append(result)
                
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(0.3)  # Rate limiting
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success(f"✅ Scan complete! Found {len(bullish_signals)} bullish and {len(bearish_signals)} bearish signals")
            
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
                        
                        # Volume info
                        st.write(f"**📊 Volume Analysis:**")
                        st.write(f"  • Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'🔥' if sig['high_volume_today'] else ''}")
                        st.write(f"  • Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'🔥' if sig['high_volume_yesterday'] else ''}")
                        
                        st.success("💡 **Recommendation:** Consider BUY with stop loss below ₹{:.2f} (SMA21)".format(sig['sma21']))
            
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
                        
                        # Volume info
                        st.write(f"**📊 Volume Analysis:**")
                        st.write(f"  • Today: {format_volume(sig['today_volume'])} ({sig['volume_ratio']:.2f}x avg) {'🔥' if sig['high_volume_today'] else ''}")
                        st.write(f"  • Yesterday: {format_volume(sig['yesterday_volume'])} ({sig['volume_ratio_yesterday']:.2f}x avg) {'🔥' if sig['high_volume_yesterday'] else ''}")
                        
                        st.error("💡 **Recommendation:** Consider SELL/SHORT with stop loss above ₹{:.2f} (SMA21)".format(sig['sma21']))
            
            if not bullish_signals and not bearish_signals:
                st.info("ℹ️ No signals found matching criteria (crossover within 5 days + high volume >1.5x)")
    
    # ==================== PORTFOLIO MANAGER ====================
    elif page == "💼 Portfolio Manager":
        st.header("Portfolio Holdings")
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    pnl = current_value - row['invested_amount']
                    pnl_pct = (pnl / row['invested_amount'] * 100)
                    
                    with st.expander(f"{row['symbol']} - {row['company_name']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quantity", f"{row['quantity']:.0f}")
                            st.metric("Avg Price", f"₹{row['avg_price']:.2f}")
                        with col2:
                            st.metric("Current Price", f"₹{current_price:.2f}")
                            st.metric("Invested", format_currency(row['invested_amount']))
                        with col3:
                            st.metric("Current Value", format_currency(current_value))
                            st.metric("P&L", format_currency(pnl), f"{pnl_pct:.2f}%")
                except Exception as e:
                    st.error(f"Error loading {row['symbol']}: {str(e)}")
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
    st.sidebar.info("💡 **Tip:** Upload Excel files once and use forever!")
    st.sidebar.caption("Unified Trading System - By Ashish Gupta")


if __name__ == "__main__":
    main()
