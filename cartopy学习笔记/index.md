# cartopy笔记


#### 简介

------

常用的地图可视化的编程工具有 MATLAB、IDL、GrADS、GMT、NCL 等。而python环境下常用的地图包有Basemap、Cartopy。此前 Python 最常用的地图包是 Basemap，然而它将于 2020 年被弃用，官方推荐使用 Cartopy 包作为替代。Cartopy 是英国气象局开发的地图绘图包，实现了 Basemap 的大部分功能，还可以通过 Matplotlib 的 API 实现丰富的自定义效果。

Cartopy 是利用 Matplotlib 来画图的，因此首先要导入 `pyplot` 模块。在 Cartopy 中，每种投影都是一个类，被存放在 `cartopy.crs` 模块中，crs 即坐标参考系统（Coordinate Reference Systems）之意。所以接着要导入这个模块。这里选取最常用的等距圆柱投影 `ccrs.PlateCarree` 作为地图投影。

Matplotlib 画图是通过调用 `Axes` 类的方法来完成的。Cartopy 创造了一个 `Axes` 的子类，`GeoAxes`，它继承了前者的基本功能，还添加了一系列绘制地图元素的方法。创建一个 `GeoAxes` 对象的办法是，在创建 axes（或 subplot）时，通过参数 `projection` 指定一个 `ccrs` 中的投影。

因此用 Cartopy 画地图的基本流程并不复杂：

- 创建画布。
- 通过指定 `projection` 参数，创建 `GeoAxes` 对象。
- 调用 `GeoAxes` 的方法画图。

#### 安装Cartopy和相关的库

------

```python
conda install -c conda-forge cartopy
```

#### 投影方式及设置

------

Cartopy提供了大量的投影方式，使用`cartopy.crs`可以调用各个投影。

```python
cartopy.crs.LambertCylindrical #调用兰勃脱投影
cartopy.crs.Mercator #调用麦卡托投影
```

#### GeoAxes的一些有用的方法

------

`GeoAxes` 有不少有用的方法，这里列举如下：

- `set_global`：让地图的显示范围扩展至投影的最大范围。例如，对 `PlateCarree` 投影的 ax 使用后，地图会变成全球的。
- `set_extent`：给出元组 `(x0, x1, y0, y1)` 以限制地图的显示范围。
- `set_xticks`：设置 x 轴的刻度。
- `set_yticks`：设置 y 轴的刻度。
- `gridlines`：给地图添加网格线。
- `coastlines`：在地图上绘制海岸线。
- `stock_img`：给地图添加低分辨率的地形图背景。
- `add_feature`：给地图添加特征（例如陆地或海洋的填充、河流等）。

#### 在地图上添加数据

------

在直接调用 `ax.plot`、`ax.contourf` 等方法在地图上添加数据之前，需要了解 Cartopy 的一个核心概念：在创建一个 `GeoAxes` 对象时，通过 `projection` 关键字指定了这个地图所处的投影坐标系，这个坐标系的投影方式和原点位置都可以被指定。但是我们手上的数据很可能并不是定义在这个坐标系下的（例如那些规整的经纬度网格数据），因此在调用画图方法往地图上添加数据时，需要通过 `transform` 关键字指定我们的数据所处的坐标系。画图过程中，Cartopy 会自动进行这两个坐标系之间的换算，把我们的数据正确投影到地图的坐标系上。下面给出一个例子：

```python
# 导入所需的库
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# 定义一个在PlateCarree投影中的方框
x = [-100.0, -100.0, 100.0, 100.0, -100.0]
y = [-60.0, 60.0, 60.0, -60.0, -60.0]

# 选取两种地图投影
map_proj = [ccrs.PlateCarree(), ccrs.Mollweide()]
data_proj = ccrs.PlateCarree()

# 创建画布以及ax
fig = plt.figure()
ax1 = fig.add_subplot(211, projection=map_proj[0])
ax1.stock_img()
ax1.plot(x, y, marker='o', transform=data_proj)
ax1.fill(x, y, color='coral', transform=data_proj, alpha=0.4)
ax1.set_title('PlateCarree')

ax2 = fig.add_subplot(212, projection=map_proj[1])
ax2.stock_img()
ax2.plot(x, y, marker='o', transform=data_proj)
ax2.fill(x, y, color='coral', transform=data_proj, alpha=0.4)
ax2.set_title('Mollweide')

plt.show()
```

<center class="half">
    <img src="/images/py_proj.png" width="300"/>
</center>

可以看到，等距圆柱投影地图上的一个方框，在摩尔威投影的地图上会向两边“长胖”——尽管这两个形状代表同一个几何体。如果不给出 `transform` 关键字，那么 Cartopy 会默认数据所在的坐标系是 `PlateCarree()`。为了严谨起见，建议在使用任何画图方法（`plot`、`contourf`、`pcolormesh` 等）时都给出 `transform` 关键字。

#### 显示自定义shp

------

使用`cartopy.io.shapereader`中的`Reader`可以读取shp文件。

```python
from cartopy.io.shapereader import Reader
reader = Reader(your_shp)
```

再通过`cartopy.feature`中的`ShapelyFeature`可以加载自己的shp特征，并设置相关属性。

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeat
proj = ccrs.PlateCarree()
feature = cfeat.ShapelyFeature(reader.geometries(), proj,
                edgecolor='k', facecolor=cfeat.COLORS['land'])
```

最后通过`add_feature`来增加以上的地图信息.

```python
ax.add_feature(feature, linewidth=1)
```

#### 为地图增加经纬度刻度

------

在 0.17 及以前的版本中，**Cartopy 仅支持为直角坐标系统（等距圆柱投影和麦卡托投影）添加刻度**，而对兰勃特投影这样的则无能为力。这里以等距圆柱投影为例

```python
# 导入Cartopy专门提供的经纬度的Formatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

map_proj = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(111, projection=map_proj)

ax.set_global()
ax.stock_img()

# 设置大刻度和小刻度
tick_proj = ccrs.PlateCarree()
ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=tick_proj)
ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=tick_proj)
ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=tick_proj)
ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=tick_proj)

# 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

plt.show()
```

<center class="half">
    <img src="/images/py_ticks.png" width="300"/>
</center>

artopy 中需要用 `GeoAxes` 类的 `set_xticks` 和 `set_yticks` 方法来分别设置经纬度刻度。这两个方法还可以通过 `minor` 参数，指定是否添上小刻度。

`set_xticks` 中的 `crs` 关键字指的是我们给出的 ticks 是在什么坐标系统下定义的，这样好换算至 ax 所在的坐标系统，原理同上一节所述。如果不指定，就很容易出现把 ticks 画到地图外的情况。除了 `set_xticks`，`set_extent` 方法同样有 `crs` 关键字，我们需要多加注意。

接着利用 Cartopy 专门提供的 Formatter 来格式化刻度的标签，使之能有东经西经、南纬北纬的字母标识。

即全球地图的最右端缺失了 0° 的标识，这是 Cartopy 内部在换算 ticks 的坐标时用到了 mod 计算而导致的，解决方法见 stack overflow 上的 [这个讨论](https://stackoverflow.com/questions/56412206/cant-show-0-tick-in-right-when-central-longitude-180)，这里就不赘述了。额外提一句，NCL 对于这种情况就能正确处理。

Cartopy 还有一个很坑的地方在于，`set_extent` 与指定 ticks 的效果会互相覆盖：如果你先用前者设置好了地图的显示范围，接下来的 `set_xticks` 超出了 extent 的范围的话，最后的出图范围就会以 ticks 的范围为准。因此使用时要注意 ticks 的范围，或把 `set_extent` 操作放在最后实施。

#### 为Lambert投影地图添加刻度

------

这里的 Lambert 投影指的是 Lambert conformal conic 投影（兰勃特等角圆锥投影），是通过让圆锥面与地球相切（割），然后将地球表面投影到圆锥面上来实现的。作为一种等角地图投影，Lambert 投影能够较好地保留区域的角度和形状，适合用于对中纬度东西方向分布的大陆板块进行制图。详细的描述请见维基和 [ArcMap 上的介绍](https://desktop.arcgis.com/zh-cn/arcmap/latest/map/projections/lambert-conformal-conic.htm)。

在 Cartopy 中，这一投影通过 `LambertConformal` 类来实现

```python
import cartopy.crs as ccrs

map_proj = ccrs.LambertConformal(
    central_longitude=105, standard_parallels=(25, 47)
)
```

这个类的参数有很多，这里为了画出中国地图，只需要指定中央经线 `central_longitude=105`，两条标准纬线 `standard_parallels=(25, 47)`，参数来源是 [中国区域Lambert&Albers投影参数](http://blog.sina.com.cn/s/blog_4aa4593d0102ziux.html) 这篇博文。

我们一般需要通过 `GeoAxes` 的 `set_extent` 方法截取我们关心的区域，下面截取 80°E-130°E，15°N-55°N 的范围

```python
extent = [80, 130, 15, 55]
ax.set_extent(extent, crs=ccrs.PlateCarree())
```

<center class="half">
    <img src="/images/py_set_extent.png" width="300"/>
</center>

道理上来说给出经纬度的边界，截取出来的应该是一个更小的扇形，但按 [issue #697](https://github.com/SciTools/cartopy/issues/697) 的说法，`set_extent` 会选出一个刚好包住这个小扇形的矩形作为边界。这里来验证一下这个说法

```python
import matplotlib as mpl
rect = mpl.path.Path([
    [extent[0], extent[2]],
    [extent[0], extent[3]],
    [extent[1], extent[3]],
    [extent[1], extent[2]],
    [extent[0], extent[2]]
]).interpolated(20)
line = rect.vertices
ax.plot(line[:, 0], line[:, 1], lw=1, c='r', transform=ccrs.Geodetic())
```

这段代码是将 `extent` 所描述的小扇形画在地图上，结果在上一张图里有。可以看到，这个小扇形确实刚好被矩形边界给包住。

如果确实需要截取出扇形的区域，可以用 `set_boundary` 方法，效果如下图

```python
ax.set_boundary(rect, transform=ccrs.Geodetic())
```

---自制方法，添加刻度---

这里尝试自己写一个添加刻度的函数。思路来自 Cartopy 的 `Gridliner` 类的源码和 [Labelling grid lines on a Lambert Conformal projection](https://gist.github.com/ajdawson/dd536f786741e987ae4e) 这篇 note。

原理是想办法在 Lambert 投影坐标系（这里亦即 Matplotlib 的 data 坐标系）下表示出 xy 轴和网格线的空间位置，若一条网格线与一个轴线相交，那么`这个交点的位置即刻度的位置`。最后直接将这些位置用于 `set_xticks` 和 `set_yticks` 方法。`判断两线相交用到了 Shapley 库`。代码和效果如下

```python
import numpy as np
import shapely.geometry as sgeom

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def find_x_intersections(ax, xticks):
    '''找出xticks对应的经线与下x轴的交点在data坐标下的位置和对应的ticklabel.'''
    # 获取地图的矩形边界和最大的经纬度范围.
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    xaxis = sgeom.LineString([(x0, y0), (x1, y0)])
    # 仅选取能落入地图范围内的ticks.
    lon_ticks = [tick for tick in xticks if tick >= lon0 and tick <= lon1]

    # 每条经线有nstep个点.
    nstep = 50
    xlocs = []
    xticklabels = []
    for tick in lon_ticks:
        lon_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.full(nstep, tick),
                np.linspace(lat0, lat1, nstep)
            )[:, :2]
        )
        # 如果经线与x轴有交点,获取其位置.
        if xaxis.intersects(lon_line):
            point = xaxis.intersection(lon_line)
            xlocs.append(point.x)
            xticklabels.append(tick)
        else:
            continue

    # 用formatter添上度数和东西标识.
    formatter = LongitudeFormatter()
    xticklabels = [formatter(label) for label in xticklabels]

    return xlocs, xticklabels

def find_y_intersections(ax, yticks):
    '''找出yticks对应的纬线与左y轴的交点在data坐标下的位置和对应的ticklabel.'''
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    yaxis = sgeom.LineString([(x0, y0), (x0, y1)])
    lat_ticks = [tick for tick in yticks if tick >= lat0 and tick <= lat1]

    nstep = 50
    ylocs = []
    yticklabels = []
    for tick in lat_ticks:
        # 注意这里与find_x_intersections的不同.
        lat_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.linspace(lon0, lon1, nstep),
                np.full(nstep, tick)
            )[:, :2]
        )
        if yaxis.intersects(lat_line):
            point = yaxis.intersection(lat_line)
            ylocs.append(point.y)
            yticklabels.append(tick)
        else:
            continue

    formatter = LatitudeFormatter()
    yticklabels = [formatter(label) for label in yticklabels]

    return ylocs, yticklabels

def set_lambert_ticks(ax, xticks, yticks):
    '''
    给一个LambertConformal投影的GeoAxes在下x轴与左y轴上添加ticks.

    要求地图边界是矩形的,即ax需要提前被set_extent方法截取成矩形.
    否则可能会出现错误.

    Parameters
    ----------
    ax : GeoAxes
        投影为LambertConformal的Axes.

    xticks : list of floats
        x轴上tick的位置.

    yticks : list of floats
        y轴上tick的位置.

    Returns
    -------
    None
    '''
    # 设置x轴.
    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels)
    # 设置y轴.
    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels)
```

需要注意的是，这个方法要求在设置刻度之前就通过 `set_extent` 方法截取出矩形的边界，否则可能有奇怪的结果。另外经测试对 Albers 投影也适用。

#### 绘制标准中国地图

------

台湾岛、钓鱼岛、南海诸岛、藏南地区、阿克赛钦地区、九段线这些典型的易错易少区域。



#### 画图的例子

------

下面举一个读取 NETCDF 格式的 ERA5 文件并画图的例子：

```python
#-------------------------------------------------------------------------
# 画出ERA5数据在500hPa高度的相对湿度和水平风场.
#-------------------------------------------------------------------------
import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def add_Chinese_provinces(ax, **kwargs):
    '''
    给一个GeoAxes添加中国省界.

    Parameters
    ----------
    ax : GeoAxes
        被绘制的GeoAxes,投影不限.

    **kwargs
        绘制feature时的Matplotlib关键词参数,例如linewidth,facecolor,alpha等.

    Returns
    -------
    None
    '''
    proj = ccrs.PlateCarree()
    shp_filepath = 'D:/Python/geos/data/Data_ipynb/bou2_4p.shp'
    reader = Reader(shp_filepath)
    provinces = cfeature.ShapelyFeature(reader.geometries(), proj)
    ax.add_feature(provinces, **kwargs)

def set_map_ticks(ax, dx=60, dy=30, nx=0, ny=0, labelsize='medium'):
    '''
    为PlateCarree投影的GeoAxes设置tick和tick label.
    需要注意,set_extent应该在该函数之后使用.

    Parameters
    ----------
    ax : GeoAxes
        需要被设置的GeoAxes,要求投影必须为PlateCarree.

    dx : float, default: 60
        经度的major ticks的间距,从-180度开始算起.默认值为10.

    dy : float, default: 30
        纬度的major ticks,从-90度开始算起,间距由dy指定.默认值为10.

    nx : float, default: 0
        经度的minor ticks的个数.默认值为0.

    ny : float, default: 0
        纬度的minor ticks的个数.默认值为0.

    labelsize : str or float, default: 'medium'
        tick label的大小.默认为'medium'.

    Returns
    -------
    None
    '''
    if not isinstance(ax.projection, ccrs.PlateCarree):
        raise ValueError('Projection of ax should be PlateCarree!')
    proj = ccrs.PlateCarree()   # 专门给ticks用的crs.

    # 设置x轴.
    major_xticks = np.arange(-180, 180 + 0.9 * dx, dx)
    ax.set_xticks(major_xticks, crs=proj)
    if nx > 0:
        ddx = dx / (nx + 1)
        minor_xticks = np.arange(-180, 180 + 0.9 * ddx, ddx)
        ax.set_xticks(minor_xticks, minor=True, crs=proj)

    # 设置y轴.
    major_yticks = np.arange(-90, 90 + 0.9 * dy, dy)
    ax.set_yticks(major_yticks, crs=proj)
    if ny > 0:
        ddy = dy / (ny + 1)
        minor_yticks = np.arange(-90, 90 + 0.9 * ddy, ddy)
        ax.set_yticks(minor_yticks, minor=True, crs=proj)

    # 为tick label增添度数标识.
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=labelsize)

if __name__ == '__main__':
    # 设置绘图区域.
    lonmin, lonmax = 75, 150
    latmin, latmax = 15, 60
    extent = [lonmin, lonmax, latmin, latmax]

    # 读取extent区域内的数据.
    filename = 't_uv_rh_gp_ERA5.nc'
    with xr.open_dataset(filename) as ds:
        # ERA5文件的纬度单调递减,所以先反转过来.
        ds = ds.sortby(ds.latitude)
        ds = ds.isel(time=0).sel(
            longitude=slice(lonmin, lonmax),
            latitude=slice(latmin, latmax),
            level=500
        )

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 添加海岸线和中国省界.
    ax.coastlines(resolution='10m', lw=0.3)
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    # 设置经纬度刻度.
    set_map_ticks(ax, dx=15, dy=15, nx=1, ny=1, labelsize='small')
    ax.set_extent(extent, crs=proj)

    # 画出相对湿度的填色图.
    im = ax.contourf(
        ds.longitude, ds.latitude, ds.r,
        levels=np.linspace(0, 100, 11), cmap='RdYlBu_r',
        extend='both', alpha=0.8
    )
    cbar = fig.colorbar(
        im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal',
        format=mpl.ticker.PercentFormatter()
    )
    cbar.ax.tick_params(labelsize='small')

    # 画出风箭头.
    # 直接使用DataArray会报错,所以转换成ndarray.
    # regrid_shape给出地图最短的那个维度要画出的风箭头数.
    # angles指定箭头角度的确定方式.
    # scale_units指定箭头长度的单位.
    # scale给出data units/arrow length units的值.scale越小,箭头越长.
    # units指定箭头维度(长度除外)的单位.
    # width给出箭头shaft的宽度.
    Q = ax.quiver(
        ds.longitude.data, ds.latitude.data,
        ds.u.data, ds.v.data,
        regrid_shape=20, angles='uv',
        scale_units='xy', scale=12,
        units='xy', width=0.15,
        transform=proj
    )
    # 在ax右下角腾出放quiverkey的空间.
    # zorder需大于1,以避免被之前画过的内容遮挡.
    w, h = 0.12, 0.12
    rect = mpl.patches.Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes,
        fc='white', ec='k', lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # 添加quiverkey.
    # U指定风箭头对应的速度.
    qk = ax.quiverkey(
        Q, X=1-w/2, Y=0.7*h, U=40,
        label=f'{40} m/s', labelpos='S', labelsep=0.05,
        fontproperties={'size': 'x-small'}
    )

    title = 'Relative Humidity and Wind at 500 hPa'
    ax.set_title(title, fontsize='medium')

    plt.show()
```


