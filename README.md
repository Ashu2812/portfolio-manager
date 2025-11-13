# Unified Stock Analysis & Portfolio Monitoring System v1.0

## 🎯 Overview

This unified system combines two powerful features into a single, streamlined solution:

1. **Market Scanner**: Identifies new trading opportunities based on SMA 9/21 crossovers, volume spikes, and news sentiment
2. **Portfolio Monitor**: Tracks your existing positions and provides data-driven exit recommendations

## ✨ Key Benefits

### Before (2 Separate Systems):
- ❌ Run 2 different Python scripts
- ❌ Manage 2 separate Excel files
- ❌ Duplicate code and functionality
- ❌ Time-consuming workflow

### After (1 Unified System):
- ✅ Single Python script to run
- ✅ One Excel file to manage
- ✅ Seamless integration
- ✅ Efficient workflow

## 📋 Excel File Format

### Required Columns (for all stocks):
- **Name**: Company name
- **Symbol**: Stock ticker symbol
- **ISIN**: BSE scrip code (optional but recommended for BSE news)

### Optional Columns (for portfolio positions only):
- **Portfolio_Price**: Your entry price
- **Portfolio_Action**: BUY or SELL
- **Portfolio_Qty**: Number of shares (optional)

### How It Works:
- **Scan-only stocks**: Leave Portfolio columns empty → System only scans for opportunities
- **Portfolio positions**: Fill in Portfolio columns → System monitors + scans

### Example:

| Name | Symbol | ISIN | Portfolio_Price | Portfolio_Action | Portfolio_Qty |
|------|--------|------|-----------------|------------------|---------------|
| Reliance Industries | RELIANCE | 500325 | | | | ← Scan only
| TCS Limited | TCS | 532540 | 3000.00 | BUY | 200 | ← Monitor + Scan
| Infosys Limited | INFY | 500209 | | | | ← Scan only
| HDFC Bank | HDFCBANK | 500180 | 1650.00 | BUY | 150 | ← Monitor + Scan

## 🚀 How to Use

### Step 1: Prepare Your Excel File

Use the provided `unified_stocks_input.xlsx` template:
- It already contains your 223 stocks for scanning
- Your 20 portfolio positions are highlighted in green
- You can add/remove stocks as needed
- Update portfolio prices and quantities regularly

### Step 2: Run the Analysis

```bash
python unified_stock_analysis.py
```

### Step 3: Choose Analysis Mode

When prompted, select:
1. **Market Scan Only** - Find new opportunities (faster)
2. **Portfolio Monitor Only** - Check your holdings (faster)
3. **Both** - Complete analysis (recommended for daily use)

### Step 4: Review Results

The system will display:

**For Market Scanning:**
- 🟢 Bullish opportunities (potential buy signals)
- 🔴 Bearish opportunities (potential short/sell signals)
- Technical details, volume analysis, and news sentiment

**For Portfolio Monitoring:**
- 🔴 EXIT NOW positions (high confidence exit signals)
- 🟡 WATCH CLOSELY positions (warning signals)
- 🟢 CONTINUE HOLDING positions (healthy positions)

### Step 5: Export Results (Optional)

Export detailed analysis to Excel with:
- Market_Opportunities sheet
- Portfolio_Monitor sheet

## 📊 Technical Analysis Criteria

### Market Scanner Criteria:
✅ SMA 9 crosses above/below SMA 21 (within last 5 days)
✅ Volume surge (1.5x+ of 21-day average)
✅ News sentiment analysis
✅ Pattern recognition

### Portfolio Exit Signals:
✅ Profit/Loss thresholds (5% profit target, -3% stop loss)
✅ Trend reversal detection (SMA crossovers)
✅ Volume confirmation (high volume against position)
✅ News sentiment shifts
✅ Confidence-based recommendations (0-100%)

## 📁 File Structure

After running the system, you'll have:

```
Your_Folder/
├── unified_stock_analysis.py          # Main Python script
├── unified_stocks_input.xlsx          # Your input file (stocks + portfolio)
└── unified_analysis_results.xlsx      # Output file (if exported)
    ├── Market_Opportunities           # New trading signals
    └── Portfolio_Monitor              # Position recommendations
```

## 🔧 Installation

### Required Python Packages:

```bash
pip install pandas yfinance openpyxl requests textblob beautifulsoup4 numpy feedparser lxml
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### System Requirements:
- Python 3.7 or higher
- Internet connection (for real-time data and news)
- Windows/Mac/Linux

## 💡 Usage Tips

### Daily Workflow:
1. **Morning**: Run complete analysis (option 3) to:
   - Find new trading opportunities
   - Check if any holdings need attention
   
2. **Before Market Close**: Run portfolio monitor (option 2) to:
   - See if any positions hit exit criteria
   - Plan for next day

3. **Weekly**: Update your Excel file:
   - Add new stocks to scan list
   - Update portfolio with new positions
   - Remove closed positions

### Best Practices:

**For Market Scanning:**
- Focus on signals with recent crossovers (1-2 days ago)
- Confirm with high volume (2x+ average = stronger signal)
- Check news sentiment aligns with technical signal
- Always use stop losses

**For Portfolio Monitoring:**
- Act on "EXIT NOW" signals promptly
- Keep close watch on "WATCH CLOSELY" positions
- Review "CONTINUE HOLDING" positions weekly
- Don't ignore negative news sentiment

**Risk Management:**
- Never invest more than 2% per trade
- Always set stop losses (suggested: 21 SMA)
- Diversify across sectors
- Confirm signals with your own research

## 📈 Understanding the Signals

### Bullish Signal Components:
- 🟢 SMA9 crosses above SMA21 = Uptrend starting
- 📊 High volume = Strong buyer interest
- 📰 Positive news = Fundamental support
- **Action**: Consider buying with stop loss below SMA21

### Bearish Signal Components:
- 🔴 SMA9 crosses below SMA21 = Downtrend starting
- 📊 High volume = Strong seller pressure
- 📰 Negative news = Fundamental weakness
- **Action**: Consider selling/shorting with stop loss above SMA21

### Portfolio Recommendations:

**EXIT NOW (Confidence 50%+)**
- Multiple exit signals detected
- Immediate action recommended
- Review position ASAP

**WATCH CLOSELY (Confidence 30-49%)**
- Warning signals present
- Monitor daily
- Prepare exit plan

**CONTINUE HOLDING (Confidence <30%)**
- Position healthy
- No significant concerns
- Keep regular monitoring

## 🆚 Comparison: Old vs New System

### Old System:
```
1. Run stock_scanner.py → Find opportunities
   Input: stocks_to_scan.xlsx (223 stocks)
   Output: Bullish/Bearish signals

2. Run portfolio_monitor.py → Check holdings
   Input: my_portfolio.xlsx (22 positions)
   Output: Exit recommendations

Total Time: ~15-20 minutes
File Management: 2 Excel files to maintain
```

### New System:
```
1. Run unified_stock_analysis.py → Everything
   Input: unified_stocks_input.xlsx (223 stocks + 22 positions)
   Output: Complete analysis (market + portfolio)

Total Time: ~10-15 minutes
File Management: 1 Excel file to maintain
```

**Time Saved**: ~40-50%
**Efficiency Gain**: ~2x
**Reduced Complexity**: Significantly

## ⚠️ Important Notes

### Data Sources:
- Stock prices: Yahoo Finance (yfinance)
- News: Google News RSS feeds
- Technical indicators: Calculated from price data

### Limitations:
- Historical data may have gaps on holidays
- News sentiment is automated (verify important news manually)
- Technical signals are not 100% accurate (use as one input)
- Past performance doesn't guarantee future results

### Disclaimer:
This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always:
- Do your own research
- Consult with financial advisors
- Understand the risks before trading
- Never invest money you can't afford to lose

## 🐛 Troubleshooting

### Common Issues:

**Issue: "No data available for stock"**
- Solution: Check if symbol is correct (add .NS for NSE, .BO for BSE)
- Example: TCS.NS or RELIANCE.NS

**Issue: "Failed to load Excel file"**
- Solution: Ensure file format matches template
- Check for empty required columns (Name, Symbol)

**Issue: "Connection timeout"**
- Solution: Check internet connection
- Run analysis during market hours for best data

**Issue: "Analysis taking too long"**
- Solution: Reduce number of stocks or run market scan and portfolio monitor separately

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify Excel file format matches template
3. Ensure all required packages are installed
4. Check Python version (3.7+)

## 🔄 Updates & Maintenance

### Regular Updates (Recommended):
- **Daily**: Update portfolio prices in Excel
- **Weekly**: Review and update stock list
- **Monthly**: Update Python packages

### Package Updates:
```bash
pip install --upgrade pandas yfinance openpyxl requests textblob beautifulsoup4 numpy feedparser lxml
```

## 📝 Changelog

### Version 1.0 (Current)
- ✅ Unified market scanner and portfolio monitor
- ✅ Single Excel input file
- ✅ Flexible column detection
- ✅ Enhanced news integration
- ✅ Confidence-based recommendations
- ✅ Excel export with multiple sheets
- ✅ Color-coded portfolio positions
- ✅ Comprehensive reporting

---

## 🎓 Quick Start Guide

### For First-Time Users:

1. **Install Python packages**:
   ```bash
   pip install pandas yfinance openpyxl requests textblob beautifulsoup4 numpy feedparser lxml
   ```

2. **Prepare your Excel file**:
   - Use `unified_stocks_input.xlsx` template
   - Fill portfolio columns for positions you own
   - Leave empty for stocks you only want to scan

3. **Run the system**:
   ```bash
   python unified_stock_analysis.py
   ```

4. **Choose option 3** (Both) for complete analysis

5. **Review results** and export to Excel if needed

### That's it! You're ready to start analyzing! 📈

---

**Remember**: This is a tool to assist your decision-making, not replace it. Always combine technical analysis with fundamental research, risk management, and your own judgment. Happy trading! 🚀
