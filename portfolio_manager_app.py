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
            raise ValueError(f"Cannot sell {quantity} shares. Only {current_qty} available.")
        
        profit_loss = (sell_price - avg_price) * quantity
        profit_loss_pct = ((sell_price - avg_price) / avg_price) * 100
        
        remaining_qty = current_qty - quantity
        if remaining_qty > 0:
            remaining_invested = remaining_qty * avg_price
            cursor.execute('''
                UPDATE holdings
                SET quantity = ?, invested_amount = ?, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ?
            ''', (remaining_qty, remaining_invested, symbol.upper()))
        else:
            cursor.execute('DELETE FROM holdings WHERE symbol = ?', (symbol.upper(),))
        
        return (symbol.upper(), holding[2], quantity, avg_price,
                sell_price, profit_loss, profit_loss_pct)
    
    def bulk_import_portfolio(self, df: pd.DataFrame) -> int:
        """Bulk import portfolio from Excel"""
        count = 0
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # Try different column name variations
                symbol = None
                for col in ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'Stock']:
                    if col in df.columns:
                        symbol = str(row.get(col, '')).strip().upper()
                        if symbol and symbol != 'NAN':
                            break
                
                if not symbol:
                    continue
                
                # Company name
                company_name = None
                for col in ['Name', 'name', 'Company', 'company', 'Stock_Name']:
                    if col in df.columns:
                        company_name = str(row.get(col, symbol))
                        if company_name and company_name != 'nan':
                            break
                if not company_name:
                    company_name = symbol
                
                # Quantity
                quantity = 0
                for col in ['Quantity', 'quantity', 'Qty', 'qty', 'QTY']:
                    if col in df.columns:
                        try:
                            quantity = float(row.get(col, 0))
                            if quantity > 0:
                                break
                        except:
                            pass
                
                # Price
                price = 0
                for col in ['Price', 'price', 'Buy_Price', 'Buy/Sell_Price', 'Portfolio_Price', 'Entry_Price', 'Rate']:
                    if col in df.columns:
                        try:
                            price = float(row.get(col, 0))
                            if price > 0:
                                break
                        except:
                            pass
                
                # Action
                action = 'BUY'
                for col in ['Action', 'action', 'Type', 'Portfolio_Action', 'Side']:
                    if col in df.columns:
                        action_val = str(row.get(col, 'BUY')).upper()
                        if action_val in ['BUY', 'SELL', 'B', 'S']:
                            action = 'BUY' if action_val in ['BUY', 'B'] else 'SELL'
                            break
                
                if quantity > 0 and price > 0:
                    # Get transaction date
                    trans_date = date.today()
                    for col in ['Date', 'date', 'Transaction_Date', 'Entry_Date']:
                        if col in df.columns:
                            try:
                                trans_date = pd.to_datetime(row.get(col)).date()
                                break
                            except:
                                pass
                    
                    self.add_transaction(symbol, company_name, action, quantity, price, trans_date, 
                                       notes="Bulk imported from Excel")
                    count += 1
                else:
                    errors.append(f"Row {idx+1}: {symbol} - Invalid quantity ({quantity}) or price ({price})")
            except Exception as e:
                errors.append(f"Row {idx+1}: {str(e)}")
                continue
        
        if errors:
            st.warning(f"⚠️ Some rows were skipped. Total imported: {count}")
            with st.expander("View errors"):
                for err in errors[:10]:  # Show first 10 errors
                    st.text(err)
        
        return count
    
    def add_to_watchlist(self, symbols: List[str]):
        """Add stocks to watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = 0
        
        for symbol in symbols:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO watchlist (symbol, company_name)
                    VALUES (?, ?)
                ''', (symbol.upper(), symbol.upper()))
                if cursor.rowcount > 0:
                    count += 1
            except:
                continue
        
        conn.commit()
        conn.close()
        return count
    
    def get_watchlist(self) -> List[str]:
        """Get all watchlist symbols"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT symbol FROM watchlist ORDER BY symbol')
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    
    def clear_watchlist(self):
        """Clear watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM watchlist')
        conn.commit()
        conn.close()
    
    def get_holdings(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM holdings ORDER BY symbol', conn)
        conn.close()
        return df
    
    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f'SELECT * FROM transactions ORDER BY transaction_date DESC LIMIT {limit}', conn)
        conn.close()
        return df
    
    def get_realized_pnl(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM realized_pnl ORDER BY created_at DESC', conn)
        conn.close()
        return df
    
    def get_portfolio_summary(self) -> Dict:
        holdings = self.get_holdings()
        realized_pnl = self.get_realized_pnl()
        
        if holdings.empty:
            return {
                'total_holdings': 0,
                'total_invested': 0,
                'total_current_value': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
                'realized_pnl': 0 if realized_pnl.empty else realized_pnl['profit_loss'].sum(),
                'total_pnl': 0 if realized_pnl.empty else realized_pnl['profit_loss'].sum()
            }
        
        total_invested = holdings['invested_amount'].sum()
        current_values = []
        
        for _, row in holdings.iterrows():
            try:
                ticker = yf.Ticker(f"{row['symbol']}.NS")
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                current_value = current_price * row['quantity']
                current_values.append(current_value)
            except:
                current_values.append(row['invested_amount'])
        
        total_current_value = sum(current_values)
        unrealized_pnl = total_current_value - total_invested
        unrealized_pnl_pct = (unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
        realized_pnl_total = 0 if realized_pnl.empty else realized_pnl['profit_loss'].sum()
        
        return {
            'total_holdings': len(holdings),
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'realized_pnl': realized_pnl_total,
            'total_pnl': unrealized_pnl + realized_pnl_total
        }


# ==================== STOCK ANALYZER ====================

def analyze_stock(symbol: str, company_name: str) -> Dict:
    """Analyze stock for SMA crossover and volume"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo")
        
        if hist.empty or len(hist) < 30:
            return None
        
        hist['SMA9'] = hist['Close'].rolling(window=9).mean()
        hist['SMA21'] = hist['Close'].rolling(window=21).mean()
        hist['Volume_21d_avg'] = hist['Volume'].rolling(window=21).mean()
        
        latest = hist.iloc[-1]
        current_price = latest['Close']
        sma9 = latest['SMA9']
        sma21 = latest['SMA21']
        
        if pd.isna(sma9) or pd.isna(sma21):
            return None
        
        current_trend = 'BULLISH' if sma9 > sma21 else 'BEARISH'
        
        # Detect crossover
        crossover_detected = False
        crossover_type = None
        crossover_day = None
        
        for i in range(1, min(6, len(hist))):
            prev = hist.iloc[-(i+1)]
            curr = hist.iloc[-i]
            
            if pd.notna(prev['SMA9']) and pd.notna(prev['SMA21']):
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
        
        # Volume analysis
        today_volume = latest['Volume']
        vol_21day_avg = latest['Volume_21d_avg']
        volume_ratio = today_volume / vol_21day_avg if vol_21day_avg > 0 else 0
        
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
            'volume_ratio': volume_ratio,
            'high_volume': volume_ratio >= 1.5
        }
    except:
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


# ==================== MAIN APP ====================

def main():
    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = PortfolioDatabase()
    
    db = st.session_state.db
    
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
            st.metric("Current Value", format_currency(summary['total_current_value']))
        with col4:
            st.metric("Unrealized P&L", format_currency(summary['unrealized_pnl']),
                     f"{summary['unrealized_pnl_pct']:.2f}%")
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Realized P&L", format_currency(summary['realized_pnl']))
        with col2:
            st.metric("📊 Unrealized P&L", format_currency(summary['unrealized_pnl']))
        with col3:
            st.metric("🎯 Total P&L", format_currency(summary['total_pnl']))
        
        st.divider()
        
        # Quick Holdings Preview
        holdings = db.get_holdings()
        if not holdings.empty:
            st.subheader("Current Holdings")
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    pnl = current_value - row['invested_amount']
                    pnl_pct = (pnl / row['invested_amount'] * 100)
                    
                    with st.expander(f"{row['symbol']} - {row['company_name']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Qty", f"{row['quantity']:.0f}")
                            st.metric("Avg", f"₹{row['avg_price']:.2f}")
                        with col2:
                            st.metric("Current", f"₹{current_price:.2f}")
                            st.metric("Invested", format_currency(row['invested_amount']))
                        with col3:
                            st.metric("Value", format_currency(current_value))
                            st.metric("P&L", format_currency(pnl), f"{pnl_pct:.2f}%")
                except:
                    pass
        else:
            st.info("📤 No holdings yet. Go to 'Upload Files' to import your portfolio!")
    
    # ==================== UPLOAD FILES ====================
    elif page == "📤 Upload Files":
        st.header("Upload Excel Files")
        
        st.markdown("""
        <div class="upload-section">
        <h3>📊 Quick Setup</h3>
        <p>Upload your Excel files once and start using the system immediately!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload Stock Watchlist
        st.subheader("1️⃣ Upload Stock Watchlist")
        st.info("📋 Upload Excel with column 'Symbol' containing stock symbols (e.g., RELIANCE, TCS, INFY)")
        
        watchlist_file = st.file_uploader("Choose stock watchlist Excel file", 
                                         type=['xlsx', 'xls'], 
                                         key="watchlist")
        
        if watchlist_file:
            try:
                df = pd.read_excel(watchlist_file)
                st.write("Preview (first 10 rows):")
                st.dataframe(df.head(10))
                
                if st.button("✅ Import Watchlist", type="primary", key="import_watch"):
                    # Extract symbols
                    symbols = []
                    for col in ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'Stock']:
                        if col in df.columns:
                            symbols = df[col].dropna().astype(str).str.strip().str.upper().tolist()
                            symbols = [s for s in symbols if s and s != 'NAN']
                            break
                    
                    if symbols:
                        count = db.add_to_watchlist(symbols)
                        st.success(f"✅ Added {count} stocks to watchlist!")
                        st.balloons()
                    else:
                        st.error("❌ Could not find 'Symbol' column in Excel file")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        # Show current watchlist
        watchlist = db.get_watchlist()
        if watchlist:
            st.success(f"✅ Current watchlist: {len(watchlist)} stocks")
            with st.expander("View watchlist"):
                st.write(", ".join(watchlist[:50]))  # Show first 50
                if len(watchlist) > 50:
                    st.info(f"... and {len(watchlist) - 50} more")
            if st.button("🗑️ Clear Watchlist", key="clear_watch"):
                db.clear_watchlist()
                st.success("Watchlist cleared!")
                st.rerun()
        
        st.divider()
        
        # Upload Portfolio
        st.subheader("2️⃣ Upload Portfolio")
        st.info("📋 Required columns: Symbol, Quantity (or Qty), Price (or Buy_Price). Optional: Name, Action, Date")
        
        portfolio_file = st.file_uploader("Choose portfolio Excel file", 
                                         type=['xlsx', 'xls'], 
                                         key="portfolio")
        
        if portfolio_file:
            try:
                df = pd.read_excel(portfolio_file)
                st.write("Preview (first 10 rows):")
                st.dataframe(df.head(10))
                
                st.write("**Detected columns:**", ", ".join(df.columns))
                
                if st.button("✅ Import Portfolio", type="primary", key="import_port"):
                    with st.spinner("Importing portfolio..."):
                        count = db.bulk_import_portfolio(df)
                        st.success(f"✅ Successfully imported {count} portfolio positions!")
                        st.balloons()
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        st.divider()
        
        # Download templates
        st.subheader("📥 Download Excel Templates")
        st.info("💡 Download these templates to see the correct format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock watchlist template
            template_stocks = pd.DataFrame({
                'Name': ['Reliance Industries', 'TCS Limited', 'Infosys'],
                'Symbol': ['RELIANCE', 'TCS', 'INFY'],
                'ISIN': ['INE002A01018', 'INE467B01029', 'INE009A01021']
            })
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                template_stocks.to_excel(writer, index=False, sheet_name='Stocks')
            
            st.download_button(
                label="📥 Download Stock Template",
                data=buffer.getvalue(),
                file_name="stock_watchlist_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Portfolio template
            template_portfolio = pd.DataFrame({
                'Name': ['Reliance Industries', 'TCS Limited'],
                'Symbol': ['RELIANCE', 'TCS'],
                'Quantity': [100, 50],
                'Price': [2500.00, 3000.00],
                'Action': ['BUY', 'BUY'],
                'Date': [date.today(), date.today()]
            })
            
            buffer2 = io.BytesIO()
            with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
                template_portfolio.to_excel(writer, index=False, sheet_name='Portfolio')
            
            st.download_button(
                label="📥 Download Portfolio Template",
                data=buffer2.getvalue(),
                file_name="portfolio_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # ==================== STOCK SCANNER ====================
    elif page == "🔍 Stock Scanner":
        st.header("Stock Market Scanner")
        st.info("🔍 Scans for SMA 9/21 crossover within last 5 days + High volume (1.5x+ average)")
        
        # Option to use watchlist or manual entry
        scan_option = st.radio("Choose scan method:", 
                              ["📋 Use Watchlist", "✍️ Manual Entry"])
        
        symbols_to_scan = []
        
        if scan_option == "📋 Use Watchlist":
            watchlist = db.get_watchlist()
            if watchlist:
                st.success(f"📊 Watchlist contains {len(watchlist)} stocks")
                symbols_to_scan = watchlist
            else:
                st.warning("⚠️ Watchlist is empty. Upload stocks in 'Upload Files' or enter manually below.")
                stock_input = st.text_area("Enter symbols (one per line)", 
                                          "RELIANCE\nTCS\nINFY", height=150)
                symbols_to_scan = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
        else:
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
                
                if result and result['crossover_detected'] and result['crossover_day'] <= 5:
                    if result['high_volume']:
                        if result['crossover_type'] == 'BULLISH':
                            bullish_signals.append(result)
                        else:
                            bearish_signals.append(result)
                
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(0.3)
            
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
                        st.write(f"**Volume:** {sig['volume_ratio']:.2f}x average")
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
                        st.write(f"**Volume:** {sig['volume_ratio']:.2f}x average")
                        st.error("💡 **Recommendation:** Consider SELL/SHORT with stop loss above ₹{:.2f} (SMA21)".format(sig['sma21']))
            
            if not bullish_signals and not bearish_signals:
                st.info("ℹ️ No signals found matching criteria (crossover within 5 days + high volume 1.5x+)")
    
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
    st.sidebar.caption("Unified Stock Scanner - by Ashish Gupta")


if __name__ == "__main__":
    main()
