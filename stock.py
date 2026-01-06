import akshare as ak
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import time
import ta
import random

def get_ticker_with_suffix(ticker):
    """
    处理股票代码，添加中国股市后缀（但akshare不需要后缀）
    """
    ticker = ticker.strip()
    if ticker.isdigit() and len(ticker) == 6:
        return ticker  # akshare直接使用6位代码
    return ticker

def get_industry_stocks(industry_name):
    """
    根据行业名称获取该行业的股票列表
    """
    try:
        # 简化：直接使用行业代码映射
        industry_map = {
            "机械设备": "BK0477",
            "半导体": "BK1036",
            "新能源": "BK0493"
        }
        industry_code = industry_map.get(industry_name)
        if not industry_code:
            print(f"未找到行业: {industry_name}，支持的行业: {list(industry_map.keys())}")
            return []
        
        # 获取行业成分股
        stocks = ak.stock_board_industry_cons_em(symbol=industry_code)
        if stocks.empty:
            return []
        # 返回股票代码列表
        return stocks['代码'].tolist()[:10]  # 限制前10只
    except Exception as e:
        print(f"获取行业股票出错: {e}")
        return []

def select_two_stocks(stock_list):
    """
    从股票列表中选择两支股票（简单随机选择）
    """
    if len(stock_list) < 2:
        return stock_list
    return random.sample(stock_list, 2)

def get_ticker_with_suffix(ticker):
    """
    处理股票代码，添加中国股市后缀（但akshare不需要后缀）
    """
    ticker = ticker.strip()
    if ticker.isdigit() and len(ticker) == 6:
        return ticker  # akshare直接使用6位代码
    return ticker

def calculate_technical_indicators(df):
    """
    计算技术指标
    """
    try:
        # 确保数据足够
        if len(df) < 20:
            return None
        
        # 计算SMA20
        df['SMA20'] = ta.trend.sma_indicator(df['收盘'], window=20)
        
        # 计算RSI14
        df['RSI14'] = ta.momentum.rsi(df['收盘'], window=14)
        
        # 计算MACD
        macd_indicator = ta.trend.MACD(df['收盘'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_hist'] = macd_indicator.macd_diff()
        
        # 获取最新值
        latest = df.iloc[-1]
        return {
            'SMA20': latest['SMA20'],
            'RSI14': latest['RSI14'],
            'MACD': latest['MACD'],
            'MACD_signal': latest['MACD_signal'],
            'MACD_hist': latest['MACD_hist'],
            'current_price': latest['收盘']
        }
    except Exception as e:
        print(f"计算技术指标出错: {e}")
        return None

def get_market_trend():
    """
    获取上证指数过去30天的涨幅
    """
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        df = ak.stock_zh_index_daily_em(symbol="sh000001", start_date=start_str, end_date=end_str)
        if df.empty or len(df) < 2:
            return 0
        recent = df['close'].iloc[-1]
        earlier = df['close'].iloc[0]
        return (recent - earlier) / earlier  # 涨幅百分比
    except Exception as e:
        print(f"获取大盘走势出错: {e}")
        return 0

def get_industry_trend(ticker):
    """
    获取股票所属行业的趋势（简化版，使用板块指数）
    """
    try:
        # 这里简化，假设使用某个行业指数，如机械设备板块
        # 实际需要根据股票查询行业
        industry_symbol = "BK0477"  # 机械设备板块，示例
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        df = ak.stock_board_industry_hist_em(symbol=industry_symbol, start_date=start_str, end_date=end_str)
        if df.empty or len(df) < 2:
            return 0
        recent = df['收盘'].iloc[-1]
        earlier = df['收盘'].iloc[0]
        return (recent - earlier) / earlier
    except Exception as e:
        print(f"获取行业趋势出错: {e}")
        return 0

def get_company_financial(ticker):
    """
    获取公司最近财报数据（简化，获取净利润）
    """
    try:
        # 获取财务摘要
        df = ak.stock_financial_abstract(symbol=ticker)
        if df.empty:
            return None
        # 查找净利润
        net_profit_row = df[df['指标'] == '净利润']
        if not net_profit_row.empty:
            # 获取最新一期
            latest = net_profit_row.iloc[0, -1]  # 最后一列
            if isinstance(latest, str):
                latest = latest.replace(',', '')
            try:
                return float(latest) if latest else None
            except ValueError:
                return None
        return None
    except Exception as e:
        print(f"获取公司财报出错: {e}")
        return None

def get_stock_data_and_predict(ticker, days=5):
    """
    获取股票历史数据，提取当前价格，并预测未来几天的价格，计算技术指标
    """
    try:
        time.sleep(1)  # 减少请求频率
        # 计算过去60天的日期（为了计算指标需要更多数据）
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # 获取历史数据
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start_str, end_date=end_str)
        if df.empty:
            raise ValueError("无法获取历史数据")
        
        # 计算技术指标
        indicators = calculate_technical_indicators(df)
        
        prices = df['收盘'].values
        current_price = prices[-1]  # 最新的价格作为当前价格
        # 使用索引作为特征
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        model = LinearRegression()
        model.fit(X, y)
        # 预测未来days天的价格
        future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
        predicted_prices = model.predict(future_X)
        return current_price, predicted_prices, indicators
    except Exception as e:
        print(f"获取数据或预测时出错: {e}")
        return None, None, None

def give_advice(current, predicted_prices, market_trend, industry_trend, financial, indicators):
    """
    根据当前价格、预测价格、大盘走势、行业趋势、财报和技术指标给出建议和持有时间
    """
    if predicted_prices is None or current is None:
        return "无法给出建议", "N/A"
    
    score = 0
    # 基本因素
    if predicted_prices[0] > current:
        score += 1  # 预测上涨
    if market_trend > 0:
        score += 1  # 大盘上涨
    if industry_trend > 0:
        score += 1  # 行业上涨
    if financial and financial > 0:
        score += 1  # 净利润为正
    
    # 技术指标
    if indicators:
        if indicators['RSI14'] < 30:
            score += 1  # 超卖
        elif indicators['RSI14'] > 70:
            score -= 1  # 超买
        if indicators['current_price'] > indicators['SMA20']:
            score += 1  # 上涨趋势
        if indicators['MACD'] > indicators['MACD_signal']:
            score += 1  # MACD买入信号
    
    if score >= 4:
        advice = "强烈建议买入"
        hold_days = next((i+1 for i, p in enumerate(predicted_prices) if p > current), len(predicted_prices))
        hold_time = f"{hold_days} 天"
    elif score >= 2:
        advice = "建议买入"
        hold_days = next((i+1 for i, p in enumerate(predicted_prices) if p > current), len(predicted_prices))
        hold_time = f"{hold_days} 天"
    elif score >= 0:
        advice = "观望"
        hold_time = "视情况而定"
    else:
        advice = "建议卖出"
        hold_time = "立即"
    
    return advice, hold_time

class SimulatedTrader:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}  # ticker: {'shares': int, 'avg_price': float}
        self.history = []  # 交易历史
    
    def buy(self, ticker, price, shares):
        cost = price * shares
        if cost > self.cash:
            print("现金不足，无法买入")
            return False
        self.cash -= cost
        if ticker in self.positions:
            total_shares = self.positions[ticker]['shares'] + shares
            total_cost = self.positions[ticker]['avg_price'] * self.positions[ticker]['shares'] + cost
            self.positions[ticker]['shares'] = total_shares
            self.positions[ticker]['avg_price'] = total_cost / total_shares
        else:
            self.positions[ticker] = {'shares': shares, 'avg_price': price}
        self.history.append(f"买入 {ticker} {shares} 股 @ {price}")
        print(f"模拟买入 {ticker} {shares} 股 @ {price}")
        return True
    
    def sell(self, ticker, price, shares):
        if ticker not in self.positions or self.positions[ticker]['shares'] < shares:
            print("持股不足，无法卖出")
            return False
        self.cash += price * shares
        self.positions[ticker]['shares'] -= shares
        if self.positions[ticker]['shares'] == 0:
            del self.positions[ticker]
        self.history.append(f"卖出 {ticker} {shares} 股 @ {price}")
        print(f"模拟卖出 {ticker} {shares} 股 @ {price}")
        return True
    
    def get_portfolio_value(self, current_prices):
        total = self.cash
        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                total += pos['shares'] * current_prices[ticker]
        return total
    
    def print_status(self, current_prices):
        print(f"现金: {self.cash:.2f}")
        print("持仓:")
        for ticker, pos in self.positions.items():
            current_price = current_prices.get(ticker, pos['avg_price'])
            value = pos['shares'] * current_price
            pnl = (current_price - pos['avg_price']) * pos['shares']
            print(f"  {ticker}: {pos['shares']} 股 @ {pos['avg_price']:.2f}, 当前价值: {value:.2f}, 盈亏: {pnl:.2f}")
        total_value = self.get_portfolio_value(current_prices)
        print(f"总资产: {total_value:.2f}")

# 全局模拟交易员
trader = SimulatedTrader()

if __name__ == "__main__":
    # 暂时硬编码测试
    industry_input = "机械设备"
    stock_list = get_industry_stocks(industry_input)
    if not stock_list:
        print("无法获取行业股票")
        exit()
    selected_stocks = select_two_stocks(stock_list)
    print(f"选择的股票: {selected_stocks}")
    
    for ticker in selected_stocks:
        print(f"\n分析股票: {ticker}")
        current, predicted_prices, indicators = get_stock_data_and_predict(ticker)
        market_trend = get_market_trend()
        industry_trend = get_industry_trend(ticker)
        financial = get_company_financial(ticker)
        advice, hold_time = give_advice(current, predicted_prices, market_trend, industry_trend, financial, indicators)
        print(f"当前股价：{current}")
        print(f"预测未来5天价格：{predicted_prices}")
        if indicators:
            print(f"技术指标 - SMA20: {indicators['SMA20']:.2f}, RSI14: {indicators['RSI14']:.2f}, MACD: {indicators['MACD']:.4f}")
        print(f"大盘走势涨幅：{market_trend:.2%}")
        print(f"行业趋势涨幅：{industry_trend:.2%}")
        print(f"公司净利润：{financial}")
        print(f"建议：{advice}")
        print(f"建议持有时间：{hold_time}")
        
        # 模拟交易
        if advice in ["强烈建议买入", "建议买入"] and ticker not in trader.positions:
            shares = int(trader.cash // (current * 100)) * 100 // 2  # 每支股票买入一半现金的整百股
            if shares > 0:
                trader.buy(ticker, current, shares)
        elif advice == "建议卖出" and ticker in trader.positions:
            shares = trader.positions[ticker]['shares']
            trader.sell(ticker, current, shares)
    
    # 显示总资产
    current_prices = {ticker: get_stock_data_and_predict(ticker)[0] for ticker in selected_stocks}
    trader.print_status(current_prices)
