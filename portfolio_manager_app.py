"""
PORTFOLIO MANAGEMENT SYSTEM - Web Application
A complete portfolio tracker with buy/sell transaction management and P&L calculations

Features:
- Add new positions (buy/sell)
- Partial booking support (sell portion of holdings)
- Automatic P&L calculation
- Real-time portfolio value tracking
- Transaction history
- Mobile-friendly interface
- Cloud deployable (Google Colab + ngrok)

Tech Stack: Streamlit + Pandas + yfinance + SQLite
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import sqlite3
import os
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit {
        color: #28a745;
        font-weight: bold;
    }
    .loss {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


class PortfolioDatabase:
    """Database handler for portfolio operations"""
    
    def __init__(self, db_path='portfolio.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Holdings table - current positions
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
        
        # Transactions table - complete history
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
        
        # Realized P&L table - booked profits/losses
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_transaction(self, symbol: str, company_name: str, trans_type: str, 
                       quantity: float, price: float, trans_date: date, notes: str = ''):
        """Add a new transaction (buy/sell)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_amount = quantity * price
        
        # Insert transaction
        cursor.execute('''
            INSERT INTO transactions (symbol, company_name, transaction_type, 
                                     quantity, price, total_amount, transaction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol.upper(), company_name, trans_type.upper(), 
              quantity, price, total_amount, trans_date, notes))
        
        transaction_id = cursor.lastrowid
        
        # Update holdings
        if trans_type.upper() == 'BUY':
            self._add_to_holdings(cursor, symbol, company_name, quantity, price)
        elif trans_type.upper() == 'SELL':
            realized_pnl = self._reduce_from_holdings(cursor, symbol, quantity, price)
            if realized_pnl:
                # Record realized P&L
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
        """Add or update holding after buy"""
        # Check if holding exists
        cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
        holding = cursor.fetchone()
        
        if holding:
            # Update existing holding with new average price
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
            # Create new holding
            invested = quantity * price
            cursor.execute('''
                INSERT INTO holdings (symbol, company_name, quantity, avg_price, invested_amount)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol.upper(), company_name, quantity, price, invested))
    
    def _reduce_from_holdings(self, cursor, symbol: str, quantity: float, 
                             sell_price: float) -> Tuple:
        """Reduce holding after sell and calculate realized P&L"""
        cursor.execute('SELECT * FROM holdings WHERE symbol = ?', (symbol.upper(),))
        holding = cursor.fetchone()
        
        if not holding:
            raise ValueError(f"No holding found for {symbol}")
        
        current_qty = holding[3]
        avg_price = holding[4]
        
        if quantity > current_qty:
            raise ValueError(f"Cannot sell {quantity} shares. Only {current_qty} available.")
        
        # Calculate realized P&L
        buy_price = avg_price
        profit_loss = (sell_price - buy_price) * quantity
        profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100
        
        # Update or delete holding
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
        
        return (symbol.upper(), holding[2], quantity, buy_price, 
                sell_price, profit_loss, profit_loss_pct)
    
    def get_holdings(self) -> pd.DataFrame:
        """Get all current holdings"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM holdings ORDER BY symbol', conn)
        conn.close()
        return df
    
    def get_transactions(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        """Get transaction history"""
        conn = sqlite3.connect(self.db_path)
        
        if symbol:
            query = f'''
                SELECT * FROM transactions 
                WHERE symbol = '{symbol.upper()}'
                ORDER BY transaction_date DESC, created_at DESC 
                LIMIT {limit}
            '''
        else:
            query = f'''
                SELECT * FROM transactions 
                ORDER BY transaction_date DESC, created_at DESC 
                LIMIT {limit}
            '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_realized_pnl(self) -> pd.DataFrame:
        """Get all realized P&L records"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM realized_pnl 
            ORDER BY created_at DESC
        ''', conn)
        conn.close()
        return df
    
    def get_portfolio_summary(self) -> Dict:
        """Get overall portfolio statistics"""
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
        
        # Get current prices
        current_values = []
        for _, row in holdings.iterrows():
            try:
                ticker = yf.Ticker(f"{row['symbol']}.NS")
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                current_value = current_price * row['quantity']
                current_values.append(current_value)
            except:
                current_values.append(row['invested_amount'])  # Fallback to invested amount
        
        total_current_value = sum(current_values)
        unrealized_pnl = total_current_value - total_invested
        unrealized_pnl_pct = (unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
        
        realized_pnl_total = 0 if realized_pnl.empty else realized_pnl['profit_loss'].sum()
        total_pnl = unrealized_pnl + realized_pnl_total
        
        return {
            'total_holdings': len(holdings),
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'realized_pnl': realized_pnl_total,
            'total_pnl': total_pnl
        }


def get_stock_info(symbol: str) -> Dict:
    """Fetch stock information from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period='1d')
        
        return {
            'name': info.get('longName', symbol),
            'current_price': hist['Close'].iloc[-1] if not hist.empty else 0,
            'change': info.get('regularMarketChangePercent', 0),
            'valid': True
        }
    except Exception as e:
        return {
            'name': symbol,
            'current_price': 0,
            'change': 0,
            'valid': False,
            'error': str(e)
        }


def format_currency(amount: float) -> str:
    """Format amount in Indian currency style"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:,.2f}"


def main():
    """Main application"""
    
    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = PortfolioDatabase()
    
    db = st.session_state.db
    
    # Header
    st.markdown('<p class="main-header">📊 Portfolio Management System</p>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Dashboard", "Add Transaction", "Holdings", 
                            "Transaction History", "Realized P&L", "Analytics"])
    
    # Dashboard Page
    if page == "Dashboard":
        st.header("📈 Portfolio Dashboard")
        
        summary = db.get_portfolio_summary()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Holdings", summary['total_holdings'])
        
        with col2:
            st.metric("Invested", format_currency(summary['total_invested']))
        
        with col3:
            st.metric("Current Value", format_currency(summary['total_current_value']))
        
        with col4:
            unrealized_pnl = summary['unrealized_pnl']
            pnl_color = "profit" if unrealized_pnl >= 0 else "loss"
            st.metric("Unrealized P&L", 
                     format_currency(unrealized_pnl),
                     f"{summary['unrealized_pnl_pct']:.2f}%")
        
        st.divider()
        
        # P&L Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Realized P&L", format_currency(summary['realized_pnl']))
        
        with col2:
            st.metric("📊 Unrealized P&L", format_currency(summary['unrealized_pnl']))
        
        with col3:
            total_pnl = summary['total_pnl']
            st.metric("🎯 Total P&L", format_currency(total_pnl))
        
        st.divider()
        
        # Holdings preview
        st.subheader("Current Holdings")
        holdings = db.get_holdings()
        
        if not holdings.empty:
            # Add current value column
            holdings_display = holdings.copy()
            holdings_display['Current Price'] = 0.0
            holdings_display['Current Value'] = 0.0
            holdings_display['P&L'] = 0.0
            holdings_display['P&L %'] = 0.0
            
            for idx, row in holdings_display.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    current_value = current_price * row['quantity']
                    pnl = current_value - row['invested_amount']
                    pnl_pct = (pnl / row['invested_amount'] * 100)
                    
                    holdings_display.at[idx, 'Current Price'] = current_price
                    holdings_display.at[idx, 'Current Value'] = current_value
                    holdings_display.at[idx, 'P&L'] = pnl
                    holdings_display.at[idx, 'P&L %'] = pnl_pct
                except:
                    pass
            
            # Display table
            display_cols = ['symbol', 'company_name', 'quantity', 'avg_price', 
                          'Current Price', 'invested_amount', 'Current Value', 'P&L', 'P&L %']
            st.dataframe(
                holdings_display[display_cols].style.format({
                    'quantity': '{:.0f}',
                    'avg_price': '₹{:.2f}',
                    'Current Price': '₹{:.2f}',
                    'invested_amount': '₹{:.2f}',
                    'Current Value': '₹{:.2f}',
                    'P&L': '₹{:.2f}',
                    'P&L %': '{:.2f}%'
                }),
                use_container_width=True
            )
        else:
            st.info("No holdings yet. Add your first transaction!")
    
    # Add Transaction Page
    elif page == "Add Transaction":
        st.header("➕ Add Transaction")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                trans_type = st.selectbox("Transaction Type", ["BUY", "SELL"])
                symbol = st.text_input("Symbol (e.g., RELIANCE)", "").upper()
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            
            with col2:
                trans_date = st.date_input("Transaction Date", value=date.today())
                price = st.number_input("Price per Share (₹)", min_value=0.01, value=100.0, step=0.01)
                notes = st.text_area("Notes (optional)", "")
            
            # Validate and show stock info
            if symbol:
                stock_info = get_stock_info(symbol)
                if stock_info['valid']:
                    st.success(f"✅ {stock_info['name']} - Current: ₹{stock_info['current_price']:.2f}")
                    company_name = stock_info['name']
                else:
                    st.warning("⚠️ Could not fetch stock info. Please verify symbol.")
                    company_name = symbol
            else:
                company_name = ""
            
            # Calculate total
            total_amount = quantity * price
            st.info(f"💰 Total Amount: {format_currency(total_amount)}")
            
            # For SELL transactions, check available quantity
            if trans_type == "SELL" and symbol:
                holdings = db.get_holdings()
                if not holdings.empty:
                    holding = holdings[holdings['symbol'] == symbol]
                    if not holding.empty:
                        available = holding.iloc[0]['quantity']
                        st.info(f"📊 Available Quantity: {available:.0f}")
                        if quantity > available:
                            st.error(f"❌ Cannot sell {quantity} shares. Only {available:.0f} available.")
                    else:
                        st.error(f"❌ No holdings found for {symbol}")
            
            submitted = st.form_submit_button("Add Transaction", type="primary")
            
            if submitted:
                if not symbol or not company_name:
                    st.error("Please enter a valid symbol")
                else:
                    try:
                        transaction_id = db.add_transaction(
                            symbol=symbol,
                            company_name=company_name,
                            trans_type=trans_type,
                            quantity=quantity,
                            price=price,
                            trans_date=trans_date,
                            notes=notes
                        )
                        st.success(f"✅ Transaction added successfully! (ID: {transaction_id})")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Holdings Page
    elif page == "Holdings":
        st.header("📦 Current Holdings")
        
        holdings = db.get_holdings()
        
        if not holdings.empty:
            # Fetch current prices and calculate P&L
            enriched_holdings = []
            
            for _, row in holdings.iterrows():
                try:
                    ticker = yf.Ticker(f"{row['symbol']}.NS")
                    hist = ticker.history(period='1d')
                    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                    current_value = current_price * row['quantity']
                    unrealized_pnl = current_value - row['invested_amount']
                    unrealized_pnl_pct = (unrealized_pnl / row['invested_amount'] * 100)
                    
                    enriched_holdings.append({
                        'Symbol': row['symbol'],
                        'Company': row['company_name'],
                        'Quantity': row['quantity'],
                        'Avg Price': row['avg_price'],
                        'Invested': row['invested_amount'],
                        'Current Price': current_price,
                        'Current Value': current_value,
                        'P&L': unrealized_pnl,
                        'P&L %': unrealized_pnl_pct
                    })
                except Exception as e:
                    enriched_holdings.append({
                        'Symbol': row['symbol'],
                        'Company': row['company_name'],
                        'Quantity': row['quantity'],
                        'Avg Price': row['avg_price'],
                        'Invested': row['invested_amount'],
                        'Current Price': 0,
                        'Current Value': 0,
                        'P&L': 0,
                        'P&L %': 0
                    })
            
            df_holdings = pd.DataFrame(enriched_holdings)
            
            # Display as cards for mobile-friendly view
            for idx, holding in enumerate(enriched_holdings):
                with st.expander(f"{holding['Symbol']} - {holding['Company']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Quantity", f"{holding['Quantity']:.0f}")
                        st.metric("Avg Price", f"₹{holding['Avg Price']:.2f}")
                    
                    with col2:
                        st.metric("Current Price", f"₹{holding['Current Price']:.2f}")
                        st.metric("Invested", format_currency(holding['Invested']))
                    
                    with col3:
                        st.metric("Current Value", format_currency(holding['Current Value']))
                        pnl_delta = f"{holding['P&L %']:.2f}%"
                        st.metric("P&L", format_currency(holding['P&L']), pnl_delta)
            
            # Summary table
            st.divider()
            st.subheader("Summary Table")
            st.dataframe(
                df_holdings.style.format({
                    'Quantity': '{:.0f}',
                    'Avg Price': '₹{:.2f}',
                    'Invested': '₹{:.2f}',
                    'Current Price': '₹{:.2f}',
                    'Current Value': '₹{:.2f}',
                    'P&L': '₹{:.2f}',
                    'P&L %': '{:.2f}%'
                }).background_gradient(subset=['P&L %'], cmap='RdYlGn', vmin=-10, vmax=10),
                use_container_width=True
            )
            
            # Download option
            csv = df_holdings.to_csv(index=False)
            st.download_button(
                label="📥 Download Holdings as CSV",
                data=csv,
                file_name=f"holdings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No holdings found. Start by adding a BUY transaction!")
    
    # Transaction History Page
    elif page == "Transaction History":
        st.header("📜 Transaction History")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_symbol = st.text_input("Filter by Symbol (optional)", "").upper()
        with col2:
            limit = st.selectbox("Show last", [50, 100, 200, 500], index=1)
        
        transactions = db.get_transactions(symbol=filter_symbol if filter_symbol else None, limit=limit)
        
        if not transactions.empty:
            # Format display
            transactions_display = transactions.copy()
            transactions_display['transaction_date'] = pd.to_datetime(transactions_display['transaction_date']).dt.strftime('%Y-%m-%d')
            
            display_cols = ['transaction_date', 'symbol', 'company_name', 'transaction_type', 
                          'quantity', 'price', 'total_amount', 'notes']
            
            st.dataframe(
                transactions_display[display_cols].style.format({
                    'quantity': '{:.0f}',
                    'price': '₹{:.2f}',
                    'total_amount': '₹{:.2f}'
                }),
                use_container_width=True
            )
            
            # Summary statistics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            buys = transactions[transactions['transaction_type'] == 'BUY']
            sells = transactions[transactions['transaction_type'] == 'SELL']
            
            with col1:
                st.metric("Total Transactions", len(transactions))
            with col2:
                st.metric("Buy Transactions", len(buys))
            with col3:
                st.metric("Sell Transactions", len(sells))
            with col4:
                total_invested = buys['total_amount'].sum()
                st.metric("Total Invested", format_currency(total_invested))
            
            # Download option
            csv = transactions_display[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Transactions as CSV",
                data=csv,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No transactions found.")
    
    # Realized P&L Page
    elif page == "Realized P&L":
        st.header("💰 Realized Profit & Loss")
        
        realized_pnl = db.get_realized_pnl()
        
        if not realized_pnl.empty:
            # Summary metrics
            total_realized = realized_pnl['profit_loss'].sum()
            avg_return = realized_pnl['profit_loss_pct'].mean()
            winners = len(realized_pnl[realized_pnl['profit_loss'] > 0])
            losers = len(realized_pnl[realized_pnl['profit_loss'] < 0])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Realized P&L", format_currency(total_realized))
            with col2:
                st.metric("Avg Return %", f"{avg_return:.2f}%")
            with col3:
                st.metric("Winning Trades", winners)
            with col4:
                st.metric("Losing Trades", losers)
            
            st.divider()
            
            # Detailed table
            realized_display = realized_pnl.copy()
            realized_display['created_at'] = pd.to_datetime(realized_display['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            display_cols = ['created_at', 'symbol', 'company_name', 'quantity', 
                          'buy_price', 'sell_price', 'profit_loss', 'profit_loss_pct']
            
            st.dataframe(
                realized_display[display_cols].style.format({
                    'quantity': '{:.0f}',
                    'buy_price': '₹{:.2f}',
                    'sell_price': '₹{:.2f}',
                    'profit_loss': '₹{:.2f}',
                    'profit_loss_pct': '{:.2f}%'
                }).background_gradient(subset=['profit_loss'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Download option
            csv = realized_display[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Realized P&L as CSV",
                data=csv,
                file_name=f"realized_pnl_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No realized P&L yet. Sell some holdings to see realized profits/losses.")
    
    # Analytics Page
    elif page == "Analytics":
        st.header("📊 Portfolio Analytics")
        
        holdings = db.get_holdings()
        transactions = db.get_transactions(limit=500)
        realized_pnl = db.get_realized_pnl()
        
        if not holdings.empty:
            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")
            
            allocation_data = holdings.copy()
            allocation_data['percentage'] = (allocation_data['invested_amount'] / 
                                            allocation_data['invested_amount'].sum() * 100)
            
            fig = px.pie(
                allocation_data,
                values='invested_amount',
                names='symbol',
                title='Holdings by Investment Amount',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction timeline
            if not transactions.empty:
                st.subheader("Transaction Timeline")
                
                trans_timeline = transactions.copy()
                trans_timeline['transaction_date'] = pd.to_datetime(trans_timeline['transaction_date'])
                trans_timeline = trans_timeline.groupby(['transaction_date', 'transaction_type']).agg({
                    'total_amount': 'sum'
                }).reset_index()
                
                fig = px.bar(
                    trans_timeline,
                    x='transaction_date',
                    y='total_amount',
                    color='transaction_type',
                    title='Buy/Sell Activity Over Time',
                    labels={'total_amount': 'Amount (₹)', 'transaction_date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # P&L distribution
            if not realized_pnl.empty:
                st.subheader("Realized P&L Distribution")
                
                fig = px.histogram(
                    realized_pnl,
                    x='profit_loss_pct',
                    nbins=20,
                    title='Distribution of Returns (%)',
                    labels={'profit_loss_pct': 'Return %', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for analytics. Start adding transactions!")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("💡 **Tip**: This app works great on mobile browsers!")
    st.sidebar.caption("Portfolio Manager v1.0")


if __name__ == "__main__":
    main()
