# -*- coding:utf-8 -*-
"""
@author: Kerwin Yang
"""
import logging, pickle, os
import pandas as pd, numpy as np
from datetime import datetime
from YourOwnDataSource import data_engine_funcs, data_engine


class Portfolio:
    def __init__(self):
        self.cash = None  # cash 当前持有的剩余资金
        self.positions = {}  # positions 当前持有的标的(包含不可卖出的标的)，dict类型，key是标的代码，value是position对象
        self.positions_value = 0  # positions_value 持仓价值
        self.portfolio_value = 0  # portfolio_value 当前持有的标的和现金的总价值
        self.capital_used = 0  # capital_used 已使用的现金  常规是负值 | 目前只计算股市操作使用的资金量，没有准确使用？？？
        self.start_fund = 100

    @property
    def returns(self):  # 账户总盈亏比率
        return self.portfolio_value / self.start_fund - 1

    @property
    def pnl(self):  # 账户总盈亏
        return self.portfolio_value - self.start_fund

    def __str__(self):
        return "<* 模拟账户 | 当前收益率:{:.2%}, 当前仓位:{:.2%}, 浮动盈亏:{:.2f}, 现金数量:{:.2f} *>".format(self.returns, (self.positions_value / self.portfolio_value), self.pnl, self.cash)

    def __repr__(self):
        return self.__str__()

    def summary(self):
        return {"portfolio_value": round(self.portfolio_value, 0), "returns": round(self.returns, 4), "cash": round(self.cash, 0), "positions_value": round(self.positions_value, 0), "positions": self.positions, "capital_used": round(self.capital_used, 0), "pnl": round(self.pnl, 0)}


class Position:
    def __init__(self, order):
        self.sid = order.symbol  # sid 标的代码
        self.enable_amount = 0  # enable_amount 可用数量 - 初始化时，当天不能卖
        self.amount = int(order.amount)  # amount 总持仓数量
        self.cost_basis = 0  # cost_basis 持仓成本价格
        self.records = []
        self.records.append((context.current_dt, "BUY", order.amount, order.price, order.transaction_cost))
        self.last_sale_price = order.price  #  最新价格

    def __str__(self):
        return "<* 股票仓 | 股票代码:{}, 持仓数量:{:.0f}, 可卖数量:{:.0f}, 成本价格:{:.2f}, 最新价格:{:.2f}, 参考盈亏比例:{:.2%} *>".format(self.sid, self.amount, self.enable_amount, self.cost_basis, self.last_sale_price, (self.last_sale_price / self.cost_basis - 1))

    def __repr__(self):
        return self.__str__()


class Order:
    def __init__(self, symbol, amount, price_on_order):
        self.price_on_order = price_on_order
        self.status = "null"
        # -------------- after for ptrade
        self.symbol = symbol
        self.amount = amount
        self.datetime = context.current_dt
        self.price = None
        self.commission = None
        self.tax = None

    @property
    def entrust_direction(self):  # 买卖方向
        return "BUY" if self.amount > 0 else "SELL"

    @property
    def transaction_cost(self):  # 交易成本
        return self.commission + self.tax

    def __str__(self):
        return "<* 指令 | 股票代码:{}, 创建时间:{}, 买卖方向:{}, 交易数量:{:.0f}, 委托时价格:{:.2f} *>".format(self.symbol, self.datetime, self.entrust_direction, abs(self.amount), self.price_on_order)

    def __repr__(self):
        return self.__str__()


class Config:
    """
    输入参数:
        start_date 开始日期，格式是 '2020-08-28'
        end_date 结束日期，格式是 '2020-08-28'，默认是最新的完整交易日
        start_fund 初始资金量，默认是100万
        frequency 回测频率，默认是 min 按分钟回测
    """

    commission_ratio = 0.0002  # commission_ratio -- 佣金费用
    tax_ratio = 0.001  # tax_ratio -- 印花税率
    min_commission = 5  # 最小佣金数量
    slippage = 0.002  # 滑点，采用0.2% 百分比法
    benchmark = "000300"  # 基准线

    def __init__(self, start_date, end_date="today", start_fund=1000000, frequency="min", para_optim={}, if_print_log=True):
        self.start_date = start_date  # 回测起始日期
        if end_date == "today":  # 回测结束日期
            self.end_date = data_engine_funcs.data.get_databse_last_trade_day()
        else:
            self.end_date = end_date
        self.frequency = frequency  # `day` (日线回测 --- 后续开发) 和 `min` (分钟线回测)
        self.start_fund = start_fund  # 初始账户，默认100万
        self.para_optim = para_optim  # 参数调优专用
        self.if_print_log = if_print_log

    def __str__(self):
        return "<* Config 配置内容: \n \n {} \n\n *>".format(self.summary())

    def __repr__(self):
        return self.__str__()

    def summary(self):
        return {"start_date": self.start_date, "start_fund": self.start_fund, "end_date": self.end_date, "frequency": self.frequency, "para_optim": self.para_optim}


class G:
    daycount = None  # 累计天数


class Blotter:
    def __init__(self):
        self.new_orders = []


class Context:
    def __init__(self):
        self.config = None
        self.blotter = None  # Blotter()  # blotter -- Blotter对象（记录）
        self.portfolio = None  # 模拟账户
        self.symbol_list = []
        self.recorded_vars = {}  # recorded_vars -- 收益曲线值
        self.daily_value = []
        self.data = {}
        self.bt_tag = "_deflault_bt_test_"
        self.para_optim = {}
        self.today = ""
        self.bt_dt_end = None
        self.bt_dt_start = None
        self.risk = None
        self.order_records = []
        self.position_records = []
        self.trade_csv = None
        self.failure_order_records = []

    def __str__(self):
        return "<* Context（tag：{}） 回测期间: {} - {}, 当前策略收益率: {:.2%},  回测开始时间: {},  回测耗时:{} *>".format(self.bt_tag, self.config.start_date, self.config.end_date, self.portfolio.returns, self.bt_dt_start, self.cal_consume_time())

    def __repr__(self):
        return self.__str__()

    def cal_consume_time(self):
        try:
            return data_engine_funcs.str_deltatime((self.bt_dt_end - self.bt_dt_start).total_seconds())
        except:
            return "回测未结束"

    def summary(self):
        return {
            "tag": self.bt_tag,
            "start_date": self.config.start_date if self.config else "None",
            "start_fund": self.config.start_fund if self.config else "None",
            "end_date": self.config.end_date if self.config else "None",
            "end_value": round(self.portfolio.portfolio_value, 2),
            "risk": self.risk,
            "config": self.config.summary() if self.config else "None",
            "last_run_day": self.today,
            "para_optim": self.para_optim,
            "portfolio": self.portfolio.summary() if self.portfolio else "None",
            "bt_start_time": self.bt_dt_start,
            "bt_time_consume": self.cal_consume_time(),
        }


def _set_log():
    logging.basicConfig(level=logging.NOTSET, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("PA-Training")
    # handler = logging.FileHandler(log_path)
    # handler.setLevel(logging.WARN)
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -  %(message)s")
    # handler.setFormatter(formatter)
    # log.addHandler(handler)
    return log


g = G()
log = _set_log()
context = Context()


class Get_his:
    minor_axis = []
    all_dict = {}

    def minor_xs(self, symbol):
        return self.all_dict[symbol]

    def __init__(self):
        self.minor_axis = []
        self.all_dict = {}


# =======Ptrade仿真函数
def set_universe(symbol_list):
    context.symbol_list = symbol_list


def get_history(count, frequency="1d", field=["open", "high", "low", "close", "volume", "money"], security_list=None, fq=None, include=False):
    if frequency == "1m":
        _gh = Get_his()
        for symbol in security_list:
            # df_min = data_engine.min(symbol[:6], _date=context.today, _type="last_three_day", his_dt=context.current_dt).rename(columns={"vol": "volume", "amount": "money"})
            df_min = data_engine.min(symbol[:6], _date=context.today, _type="last_three_day").rename(columns={"vol": "volume", "amount": "money"})
            _gh.all_dict[symbol] = df_min
        _gh.minor_axis = _gh.all_dict.keys()
        return _gh
    elif frequency == "1d":
        _gh = Get_his()
        for symbol in security_list:
            _gh.all_dict[symbol] = data_engine.day(symbol[:6], context.today, qfq=True, if_include=False).rename(columns={"vol": "volume", "amount": "money"})
        _gh.minor_axis = _gh.all_dict.keys()
        return _gh


def order(symbol, amount, limit_price=None):
    price_on_order = context.data[symbol]["close"]  # 回测以市价类型进行委托。
    _order = Order(symbol, amount, price_on_order)
    context.blotter.new_orders.append(_order)
    return None


def order_market(symbol, amount, limit_price=None):
    return order(symbol, amount, limit_price)


def order_value(symbol, value, limit_price=None):
    price_on_order = context.data[symbol]["close"]
    amount = int(value / price_on_order / 100) * 100  # 封装，调整买入数量
    return order(symbol, amount, limit_price)


def get_stock_name(stocks):
    _d = {}
    for _i in stocks:
        _d[_i] = data_engine.get_name_by_code(_i[:6], context.today)
    return _d


def get_stock_status(stocks, query_type="HALT", query_date=None):
    #     获取指定日期股票的ST、停牌、退市属性，参数：
    # stocks：str或list类型，例如 ['000001.SZ','000003.SZ']。该字段必须输入，否则返回None；
    # query_type：str类型，支持以下三种类型属性的查询，默认为'ST'；
    # 具体支持输入的字段包括：
    # 'ST' – 查询是否属于ST股票
    # 'HALT' – 查询是否停牌
    # 'DELISTING' – 查询是否退市
    # query_date：str类型，格式为YYYYmmdd，默认为None,表示当前日期（回测为回测当前周期，研究与交易则取系统当前时间）；
    # 返回dict类型，每支股票对应的值为True或False，当没有查询到相关数据或者输入有误时返回None；
    _d = {}
    for _i in stocks:
        if data_engine.check_if_trading(_i[:6], context.today):
            _d[_i] = False
        else:
            _d[_i] = True
    return _d


# =======券商仿真函数
def broker_process_per_min(context, data):
    # 每分钟盘前处理，获得当前分钟的bar数据，但是处理上一分钟的交易订单。模仿券商交易系统和程序。
    # 注意处理开盘09:30(没有订单，只是更新一下价格)
    # 中午11:30（接受下单），13:00 （没有处置，交易处置都在11:30）
    # 收盘15:00时间 可以完成之前的订单，但是不能有新的订单。
    # 按数据data的 min bar 进行处理。
    # 遍历new_orders， 进行成交操作
    for _i in range(len(context.blotter.new_orders) - 1, -1, -1):  # 遍历交易订单，处理订单。 假设所有都是市场订单
        order = context.blotter.new_orders[_i]
        order.filled_time = context.current_dt
        if order.amount > 0:  # 处理买单
            # 买入成交价在1、当分钟成交中的最高价、2、滑点计算价格 两者中取最小值
            _price = min(data[order.symbol]["high"], order.price_on_order * (1 + context.config.slippage))
            _money = _price * order.amount
            _commission = max(_money * context.config.commission_ratio, context.config.min_commission)
            # 税费佣金 买入税费为0
            order.price = _price  # 成交价
            order.commission = _commission  # 佣金
            order.tax = 0  # 税费
            if context.portfolio.cash < _money + _commission:  # 买入订单废单
                order.status = "cancel_by_money_not_enough"
                context.failure_order_records.append(order)
                del context.blotter.new_orders[_i]
            else:  # 买入订单成交
                context.portfolio.cash += -_money - _commission
                context.portfolio.capital_used += _money + _commission
                if order.symbol not in context.portfolio.positions:  # 建仓情况
                    _position = Position(order)
                    _position.cost_basis = (_money + _commission) / order.amount
                    context.portfolio.positions[order.symbol] = _position
                else:  # 加仓情况
                    _position = context.portfolio.positions[order.symbol]
                    _position.cost_basis = (_position.cost_basis * _position.amount + _money + _commission) / (_position.amount + order.amount)
                    _position.amount += int(order.amount)
                    _position.records.append((context.current_dt, "BUY", order.amount, order.price, order.transaction_cost))
                order.status = "fullfill_buy"
                context.on_trade_response(context, order)
                context.order_records.append(order)
                _make_trade_csv(context, order)
                del context.blotter.new_orders[_i]

        elif order.amount < 0:  # 处理卖单
            # 成交价在1、当分钟成交中最低价和2、滑点计算价格间取最大值
            _price = max(data[order.symbol]["low"], order.price_on_order * (1 - context.config.slippage))
            _money = abs(_price * order.amount)
            _commission = max(_money * context.config.commission_ratio, context.config.min_commission)
            # 税费佣金
            _tax = _money * context.config.tax_ratio
            order.price = _price  # 成交价
            order.commission = _commission  # 佣金
            order.tax = _tax  # 税费
            context.portfolio.cash += _money - _commission - _tax
            # context.portfolio.capital_used += -context.portfolio.positions[order.symbol].?_money - _commission - _tax
            _position = context.portfolio.positions[order.symbol]
            if int(-order.amount) == _position.amount:  # 清仓情况
                _position.records.append((context.current_dt, "SELL", order.amount, order.price, order.transaction_cost))
                context.position_records.append(_position)
                del context.portfolio.positions[order.symbol]
            else:  # 减仓情况
                _position.cost_basis = (_position.cost_basis * _position.amount - _money + _commission) / (_position.amount + order.amount)
                _position.amount += order.amount  # 因为负值，所以没有负号
                _position.records.append((context.current_dt, "SELL", order.amount, order.price, order.transaction_cost))
            order.status = "fullfill_sell"
            context.on_trade_response(context, order)
            context.order_records.append(order)
            _make_trade_csv(context, order)
            del context.blotter.new_orders[_i]

    #  更新portfolio  更新每只持仓股票最新价格和持仓股票总价值和模拟仓位总价值
    positions_value = 0
    _postions = context.portfolio.positions
    for symbol in _postions:  # 遍历仓位，更新股票持仓价格， 计算仓位总价值
        try:
            _postions[symbol].last_sale_price = data[symbol]["close"]
        except:
            pass
        positions_value += _postions[symbol].last_sale_price * _postions[symbol].amount
    context.portfolio.positions_value = positions_value  # 仓位总价值
    context.portfolio.portfolio_value = context.portfolio.positions_value + context.portfolio.cash  # 账户总价值


def _make_trade_csv(context, order):
    _name = get_stock_name([order.symbol])[order.symbol]
    _data = {
        "datetime": context.current_dt,
        "symbol": order.symbol,
        "name": _name,
        "entrust_direction": order.entrust_direction,
        "amount": order.amount,
        "price": round(order.price, 2),
        "money": round(abs(order.amount * order.price), 2),
        "commision": round(order.commission, 2),
        "tax": round(order.tax, 2),
        "cash": round(context.portfolio.cash, 2),
        "portfolio_value": round(context.portfolio.portfolio_value, 2),
    }
    context.trade_csv = context.trade_csv.append(_data, ignore_index=True)


def broker_after_trading_end(context, data):
    # 每日收盘后，先与收盘处理函数 after_trading_end 之前，模仿券商交易系统和程序，整理每日交易账户，删除过期的订单。
    context.blotter.new_orders = []
    positions_value = 0
    _postions = context.portfolio.positions
    for symbol in _postions:
        close_price = data_engine.price_bfq(symbol[:6], context.today)
        if close_price:
            _postions[symbol].last_sale_price = close_price
        positions_value += _postions[symbol].last_sale_price * _postions[symbol].amount
        _postions[symbol].enable_amount = _postions[symbol].amount  # t+1 更新股票可卖数量
    context.portfolio.positions_value = positions_value
    _portfolio_value = context.portfolio.positions_value + context.portfolio.cash
    context.portfolio.portfolio_value = _portfolio_value
    context.recorded_vars[context.today] = _portfolio_value  # 记录每日净值日志
    context.daily_value.append(_portfolio_value)  # 记录每日净值曲线
    _save_con(context)


def broker_process_dividend_allotted(context, data):
    _positions = context.portfolio.positions
    for symbol in _positions:
        dividend_dict = data_engine.dividend(symbol[:6], context.today)
        if dividend_dict:
            fenhong_money = _positions[symbol].amount * dividend_dict["fenhong"]
            if fenhong_money > 1:
                context.portfolio.cash += fenhong_money  # 分红
                _positions[symbol].cost_basis = (_positions[symbol].cost_basis * _positions[symbol].amount - fenhong_money) / _positions[symbol].amount
            songzhuangu_numb = int(_positions[symbol].amount * dividend_dict["songzhuangu"])
            if songzhuangu_numb > 1:
                _positions[symbol].amount += songzhuangu_numb  # 送转股
                _positions[symbol].cost_basis = _positions[symbol].cost_basis / (1 + dividend_dict["songzhuangu"])
            peigu_money = _positions[symbol].amount * dividend_dict["peigu"] * dividend_dict["peigujia"]  # 配股
            if peigu_money > 9:
                if peigu_money < context.portfolio.cash:
                    amount_old = _positions[symbol].amount
                    _positions[symbol].amount += _positions[symbol].amount * dividend_dict["peigu"]
                    amount_new = _positions[symbol].amount
                    context.portfolio.cash -= peigu_money
                    _positions[symbol].cost_basis = (_positions[symbol].cost_basis * amount_old + peigu_money) / amount_new
                else:
                    log.warning("日期：{}持仓股票中{} 进行了配股, 因为账户可用资金不足，配股失败.".format(context.today, symbol))


def _save_con(context):
    path_dir = context.path_dir
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    path = "{}/con_{}.pickle".format(path_dir, context.bt_dt_start).replace(" ", "_").replace(":", "")
    with open(path, "wb") as f:
        pickle.dump(context, f)

    pass


def summary_back_test(context):
    _pre_day = data_engine_funcs.dt.cal_trade_day(context.config.start_date, -1)
    _end_day = context.config.end_date
    benchmark_vars = data_engine.index_day(context.config.benchmark)[_pre_day:_end_day]["close"].tolist()
    context.risk = data_engine_funcs.Risk(context.daily_value, benchmark_vars)
    _save_con(context)
    path_csv = "{}/trade_bt_{}.csv".format(context.path_dir, context.bt_dt_start).replace(" ", "_").replace(":", "")
    # context.trade_csv.round = context.trade_csv.round({"price": 2, "money": 0, "commision": 2, "tax": 2, "cash": 0, "portfolio_value": 0})
    context.trade_csv.to_csv(path_csv)

    import pyecharts.options as opts
    from pyecharts.charts import Bar, Line, Page
    from pyecharts.components import Table
    from pyecharts.options import ComponentTitleOpts

    date_list = data_engine_funcs.dt.get_dates_list(_pre_day, _end_day)

    dict_bt_vars = context.recorded_vars
    dict_hs300 = data_engine.index_day(context.config.benchmark)[_pre_day:_end_day]["close"]
    dict_szzs = data_engine.index_day("000001")[_pre_day:_end_day]["close"]
    dict_cyb = data_engine.index_day("399006")[_pre_day:_end_day]["close"]

    list_bt_line = []
    list_hs300_line = []
    list_szzs_line = []
    list_cyb_line = []

    for _date in date_list:
        list_bt_line.append(dict_bt_vars[_date] / dict_bt_vars[_pre_day] - 1)
        list_hs300_line.append(dict_hs300[_date] / dict_hs300[_pre_day] - 1)
        list_szzs_line.append(dict_szzs[_date] / dict_szzs[_pre_day] - 1)
        list_cyb_line.append(dict_cyb[_date] / dict_cyb[_pre_day] - 1)

    headers = ["回测名称", "回测总收益率", "年化收益率", "波动率", "最大回测", "Alpha比率", "Beta比率", "日胜基准比率", "信息比率", "夏普比率", "索提诺比率", "下行波动率"]

    def cal_rows(strategy_name, benchmark_name, r):
        return [
            [strategy_name, data_engine_funcs.str_pct(r.strategy_total_return), data_engine_funcs.str_pct(r.strategy_annual_return), data_engine_funcs.str_pct(r.strategy_volatility), data_engine_funcs.str_pct(r.strategy_max_drawdown), round(r.alpha, 2), round(r.beta, 2), data_engine_funcs.str_pct(r.daily_win_ratio), round(r.info_ratio, 2), round(r.strategy_sharpe, 2), round(r.strategy_sortino, 2), round(r.strategy_downside_risk, 2),],
            [benchmark_name, data_engine_funcs.str_pct(r.benchmark_total_return), data_engine_funcs.str_pct(r.benchmark_annual_return), data_engine_funcs.str_pct(r.benchmark_volatility), data_engine_funcs.str_pct(r.benchmark_max_drawdown), "-", "-", "-", "-", round(r.benchmark_sharpe, 2), round(r.benchmark_sortino, 2), round(r.benchmark_downside_risk, 2),],
        ]

    table1 = Table()
    risk1 = context.risk
    table1.add(headers, cal_rows("本次回测", "沪深300", risk1))
    table1.set_global_opts(title_opts=ComponentTitleOpts(title="本次回测-沪深300", subtitle=""))

    table2 = Table()
    benchmark_szzs = data_engine.index_day("000001")[_pre_day:_end_day]["close"].tolist()
    risk2 = data_engine_funcs.Risk(context.daily_value, benchmark_szzs)
    table2.add(headers, cal_rows("本次回测", "上证指数基准", risk2))
    table2.set_global_opts(title_opts=ComponentTitleOpts(title="本次回测-上证指数", subtitle=""))

    table3 = Table()
    benchmark_cyb = data_engine.index_day("399006")[_pre_day:_end_day]["close"].tolist()
    risk2 = data_engine_funcs.Risk(context.daily_value, benchmark_cyb)
    table3.add(headers, cal_rows("本次回测", "创业板指数基准", risk2))
    table3.set_global_opts(title_opts=ComponentTitleOpts(title="本次回测-创业板指数", subtitle=""))

    x_data = date_list
    y1_data = [round(x, 2) for x in list_bt_line]
    y2_data = [round(x, 2) for x in list_hs300_line]
    y3_data = [round(x, 2) for x in list_szzs_line]
    y4_data = [round(x, 2) for x in list_cyb_line]

    line = Line().add_xaxis(x_data).add_yaxis("回测净值", y1_data, label_opts=opts.LabelOpts(is_show=False)).add_yaxis("沪深300", y2_data, label_opts=opts.LabelOpts(is_show=False)).add_yaxis("上证指数", y3_data, label_opts=opts.LabelOpts(is_show=False)).add_yaxis("创业板指数", y4_data, label_opts=opts.LabelOpts(is_show=False))

    page = Page(layout=Page.SimplePageLayout).add(table1, table2, table3, line).render("{}/sum_{}.html".format(context.path_dir, context.bt_dt_start).replace(" ", "_").replace(":", ""))


class Funcs:
    def _pass(self, arg1=None, arg2=None):
        pass

    def __init__(self, initialize, handle_data, before_trading_start=None, after_trading_end=None, on_trade_response=None, finished_back_test=None):
        self.initialize = initialize
        self.handle_data = handle_data
        self.before_trading_start = before_trading_start if before_trading_start else self._pass
        self.after_trading_end = after_trading_end if after_trading_end else self._pass
        self.on_trade_response = on_trade_response if on_trade_response else self._pass
        self.finished_back_test = finished_back_test if finished_back_test else self._pass


def run(config, funcs):
    log.info("\n\n========================\n回测开始：")
    # 配置回测环境
    context.bt_dt_start = datetime.now()
    bt_dates_list = data_engine_funcs.dt.get_dates_list(config.start_date, config.end_date)
    bt_time_list = data_engine_funcs.cons.TRADING_TIMES
    context.path_dir = "{}/pac/log/{}".format(data_engine_funcs.path.dir_lab, context.bt_tag)
    context.config = config
    context.portfolio = Portfolio()
    context.portfolio.start_fund = config.start_fund
    context.blotter = Blotter()
    context.trade_csv = pd.DataFrame(columns=["datetime", "symbol", "name", "entrust_direction", "amount", "price", "money", "commision", "tax", "cash", "portfolio_value"])
    context.trade_csv.index.name = "trade_id"
    context.para_optim = config.para_optim
    context.on_trade_response = funcs.on_trade_response
    context.portfolio.cash = config.start_fund
    context.recorded_vars[data_engine_funcs.dt.cal_trade_day(config.start_date, -1)] = config.start_fund
    context.daily_value.append(config.start_fund)
    _if_print_log = config.if_print_log

    # 初始化模块
    context.current_dt = datetime.strptime(config.start_date + " 09:20:00", "%Y-%m-%d %H:%M:%S")
    funcs.initialize(context)
    if _if_print_log:
        log.info("回测初始化initialize()函数执行完毕。")
    for bt_date in bt_dates_list:  #  回测按日期遍历
        # day loop ===========================================================================
        # 每日盘前：
        context.today = bt_date
        context.current_dt = datetime.strptime(bt_date + " 09:20:00", "%Y-%m-%d %H:%M:%S")
        data = {}
        funcs.before_trading_start(context, data)  # 盘前函数中的data是空，模拟ptrade
        if _if_print_log:
            log.info(bt_date + "的盘前before_trading_start()函数执行完毕。")
        _data_all = {}  # 配置好_data_all
        for symbol in context.symbol_list:
            _data_all[symbol] = data_engine.min(symbol[:6], bt_date)
        symbol_list_str = "-".join(context.symbol_list)  # 用于检查股票池变动
        # min loop --------------------------------------------------------------------------
        for bt_time in bt_time_list:  # 交易时间：
            context.current_dt = datetime.strptime(bt_date + " " + bt_time, "%Y-%m-%d %H:%M:%S")
            _symbol_list = context.symbol_list
            if "-".join(_symbol_list) != symbol_list_str:  # 实现检查股票池变动并更新进DATA数据中
                for symbol in _symbol_list:
                    if symbol not in _data_all:
                        _data_all[symbol] = data_engine.min(symbol[:6], bt_date)
                symbol_list_str = "-".join(_symbol_list)
            data = {}  # 按分钟位置刷新生成 data
            for symbol in _symbol_list:
                _da_s = _data_all[symbol]
                if _da_s == {}:
                    data[symbol] = {}
                else:
                    data[symbol] = _da_s[bt_time]
            context.data = data
            broker_process_per_min(context, data)  # 券商环节：处理上一分钟的交易订单，刷新上一分钟的股票价格和状态
            if bt_time != "15:00:00":  # 交易真实场景，下午3点时候挂单无效，操作无意义。
                funcs.handle_data(context, data)
        # min loop --------------------------------------------------------------------------
        # 每日盘后 ：
        context.current_dt = datetime.strptime(bt_date + " 16:00:00", "%Y-%m-%d %H:%M:%S")
        # 根据交易所规则，每天结束时会取消所有未完成交易，更新账户仓位信息
        broker_after_trading_end(context, data)
        # 盘后处理程序
        funcs.after_trading_end(context, data)
        if _if_print_log:
            str_protfolio = str(context.portfolio)
            log.info("{}的盘后after_trading_end()函数执行完毕。投资组合情况如下：\n{}\n".format(bt_date, str_protfolio))
        # 根据每只股票每日息权日数据，对股票进行息权处置
        broker_process_dividend_allotted(context, data)  # 券商环节： 处理当日盘后派息增股
        # day loop =============================================================================
    funcs.finished_back_test(context)
    context.bt_dt_end = datetime.now()
    # 回测总结整理：
    summary_back_test(context)
    log.info("\n回测结束。\n========================\n\n")

