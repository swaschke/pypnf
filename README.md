# pyPnF

A Python package for Point and Figure Charting


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

ts = dataset()

pnf = PointFigureChart(ts=ts,method='cl', reversal=1, boxsize=1, scaling='abs')
print(pnf)
```



<hr>

<h3>Literature</h3>

**Weber, Zieg (2003)** The Complete Guide to Point-and-Figure Charting - The new science of an old art. <i>Harriman House</i>, ISBN: 1-897-5972-82<br>

**du Plessis (2012)** The Definitive Guide to Point and Figure: A Comprehensive Guide to the Theory and Practical Use of the Point and Figure Charting Method. 2nd Edition. <i>Harriman House</i>, ISBN: 978-0857192455<br> 

**du Plessis (2015)** 21st Century Point and Figure - New and advanced techniques for using Point and Figure charts. <i>Harriman House</i>, ISBN: 978-0857194428<br>

**Shah (2018)** Trading the Markets the Point & Figure way : become a noiseless trader and achieve consistent success in markets. <i>Notion Press</i>, ISBN:  978-1642492248<br> 

<hr>

<h3>Copyright</h3>
pyPnF is licensed under a <a href="https://choosealicense.com/licenses/mit/">MIT License</a>.

