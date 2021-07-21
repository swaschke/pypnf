
![logo](https://swaschke.github.com/images/logo.png)

###A Python Package for Point & Figure Charting

[![license](https://img.shields.io/github/license/swaschke/pypnf)](#license)
[![Python Version](https://img.shields.io/pypi/pyversions/pypnf?style=flat)](https://pypi.org/project/pypnf/)
[![PyPi Version](https://img.shields.io/pypi/v/pypnf?style=flat)](https://pypi.org/project/pypnf/)
[![Package Status](https://img.shields.io/pypi/status/pypnf?style=flat)](https://pypi.org/project/pypnf/)

<hr>

### Installation
```python
pip install pypnf
```

<hr>

### Usage

#### Quickstart using integrated time-series example


```python
from pypnf import PointFigureChart
from pypnf import dataset

symbol = 'AAPL'  # or 'MSFT'

ts = dataset(symbol)

PnF = PointFigureChart(ts=ts, method='cl', reversal=2, boxsize=5, scaling='abs', title=symbol)

print(PnF)
```

```
Point & Figure (abs|cl) 5 x 2 | AAPL
---  -  -  -  -  -  -  -  ---
135  .  .  .  .  .  .  X  135
130  .  .  X  .  .  .  X  130
125  .  .  X  O  .  .  X  125
120  .  .  X  O  X  .  X  120
115  .  .  X  O  X  O  X  115
110  .  .  X  O  .  O  .  110
105  .  .  X  .  .  .  .  105
100  .  .  X  .  .  .  .  100
 95  .  .  X  .  .  .  .   95
 90  .  .  X  .  .  .  .   90
 85  .  .  X  .  .  .  .   85
 80  X  .  X  .  .  .  .   80
 75  X  O  X  .  .  .  .   75
 70  .  O  X  .  .  .  .   70
 65  .  O  X  .  .  .  .   65
 60  .  O  .  .  .  .  .   60
---  -  -  -  -  -  -  -  ---
printed 7/7 columns.
```


#### Quickstart using time-series data loaded via the external package yfinance

If the yfinance and/or pandas package is not yet installed use:

```python
pip install pandas
pip install yfinance
```

In order to process the downloaded data with the PointFigureChart-class the time-series data needs to be  prepared.

```python
import yfinance as yf

symbol = 'AAPL'

data = yf.Ticker(symbol)
ts = data.history(start='2018-01-01', end='2018-06-30')

# reset index
ts.reset_index(level=0, inplace=True)

# convert pd.timestamp to string
ts['Date'] = ts['Date'].dt.strftime('%Y-%m-%d')

# select required keys
ts = ts[['Date','Open','High','Low','Close']]

# convert DataFrame to dictionary
ts = ts.to_dict('list')
```

Initiate the PointFigureChart object with the prepared data and chart parameter, 
get the trendlines and print the resulting Point and Figure Chart.

```python
from pypnf import PointFigureChart

pnf = PointFigureChart(ts=ts, method='cl', reversal=3, boxsize=2, scaling='abs', title='AAPL')
pnf.get_trendlines()
print(pnf)
```

```
Point & Figure (abs|cl) 2 x 3 | AAPL
--  -  -  -  -  -  -  -  -  -  -  -  --
90  .  .  .  .  .  .  .  .  .  .  X  90
88  .  .  .  .  .  .  .  .  .  .  X  88
86  .  .  .  .  .  .  .  .  .  .  X  86
84  .  .  .  .  .  .  .  .  .  .  X  84
82  .  .  .  .  .  .  .  .  .  .  X  82
80  .  .  .  .  X  .  .  .  .  .  X  80
78  .  .  .  .  X  O  .  .  .  .  X  78
76  .  .  .  .  X  O  .  .  .  .  X  76
74  .  .  .  .  X  O  X  .  .  .  X  74
72  .  .  .  .  X  O  X  O  .  .  X  72
70  .  .  .  .  X  O  X  O  .  .  X  70
68  .  .  .  .  X  O  .  O  X  .  X  68
66  .  .  .  .  X  .  .  O  X  O  X  66
64  .  .  .  .  X  .  .  O  X  O  X  64
62  .  .  .  .  X  .  .  O  .  O  X  62
60  .  .  .  .  X  .  .  .  .  O  X  60
58  *  .  .  .  X  .  .  .  .  O  X  58
56  X  *  .  .  X  .  .  .  .  O  .  56
54  X  O  *  .  X  .  .  .  .  .  .  54
52  X  O  .  *  X  .  .  .  .  .  *  52
50  X  O  X  .  X  .  .  .  .  *  .  50
48  X  O  X  O  X  .  .  .  *  .  .  48
46  X  O  X  O  X  .  .  *  .  .  .  46
44  X  O  X  O  .  .  *  .  .  .  .  44
42  X  O  X  .  .  *  .  .  .  .  .  42
40  X  O  X  .  *  .  .  .  .  .  .  40
38  .  O  X  *  .  .  .  .  .  .  .  38
36  .  O  *  .  .  .  .  .  .  .  .  36
--  -  -  -  -  -  -  -  -  -  -  -  --
last trendline: bullish support line of length 10
printed 11/11 columns.
```

<hr>

<h3>Literature</h3>

**Weber, Zieg (2003)** The Complete Guide to Point-and-Figure Charting - The new science of an old art. <i>Harriman House</i>, ISBN: 1-897-5972-82<br>

**du Plessis (2012)** The Definitive Guide to Point and Figure: A Comprehensive Guide to the Theory and Practical Use of the Point and Figure Charting Method. 2nd Edition. <i>Harriman House</i>, ISBN: 978-0857192455<br> 

**du Plessis (2015)** 21st Century Point and Figure - New and advanced techniques for using Point and Figure charts. <i>Harriman House</i>, ISBN: 978-0857194428<br>

**Shah (2018)** Trading the Markets the Point & Figure way : become a noiseless trader and achieve consistent success in markets. <i>Notion Press</i>, ISBN:  978-1642492248<br> 

<hr>

<h3>Copyright</h3>
pyPnF is licensed under a GNU General Public License v2 (GPLv2).

