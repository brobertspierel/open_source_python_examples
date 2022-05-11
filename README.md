# Open Source Python Examples 

## Rasters

### GDAL 

	- Command line vs Python implementation 
	- Virtual rasters 
	- Clipping? 

### Numpy 
	- Array based operations vs for loops 
	- Masking
	- Multi-dimensional arrays? 

### Rasterio 
	- Window-based operations 
	- GIS-like functionality (clipping)

## Vectors 

### Pandas 
	- General tabular data 
	- Possible to use in conjunction with geospatial data (e.g. joins, merges etc.)

### Geopandas 
	- Very easy manipulation of geospatial datasets 
	- Comes with many tools that are akin to GIS GUIs

### OGR
	- Alternative to Geopandas, can be faster
	- Functionality a bit more verbose 

## Other 

### Matplotlib 
	- Standard plotting library, a lot of functionality and basis for many other things 

### Seaborn 
	- Good for more complex data or multi-dimensional plots 

### Google Earth Engine API 

	- Good for generating a large number of tasks 
	- Integration with AI/ML/DL e.g., pytorch, tensorflow


# Library import examples

```
import os 
import sys
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import glob
import pickle 
from osgeo import gdal, ogr, osr 
import subprocess
import matplotlib.pyplot as plt 
import rasterio 
import seaborn as sns 
```


# Code Examples

## Rasters 

### GDAL 

Command line implementation from [GDAL](https://gdal.org/programs/gdal_translate.html): 
`gdal_translate -of GTiff -co "TILED=YES" utm.tif utm_tiled.tif`

Command line (type) implementation: 
```
def clip_rasters_to_shapefile(raster_list,input_shapefile,output_dir,fn_modifier,noData=0): 
	"""Just what it sounds like."""
	child_dir = os.path.split(input_shapefile)[1][:-4] #get just the shapefile name with no extension
	dir_name = os.path.join(output_dir,child_dir)
	cmds=[]
	for raster in raster_list: 
		output_filename = os.path.join(output_dir,f'{os.path.split(raster)[1][:-4]}_{fn_modifier}_clipped.tif')
		print(output_filename)
		if not os.path.exists(output_filename): 
			cmd = f'gdalwarp -cutline {input_shapefile} -crop_to_cutline {raster} {output_filename} -co COMPRESS=LZW -srcnodata {noData} -dstnodata {noData}'
			#uncomment to run in parallel 
			cmds.append(cmd)
		else: 
			print(f'The {output_filename} file already exists, passing...')
	return cmds

```

Python based implementation: 
```
def gdal_raster_to_vector(input_file,output_dir,model='',**kwargs): 
	"""Converts a geotiff to a shp file. Use in a for loop to convert a directory of raster to vectors using GDAL. 
	Does not do any additional post-processing, just does a straight conversion."""
	 
	try: 
		year = re.findall('(\d{4})', os.path.split(input_file)[1])[0] 
	except IndexError: 
		year = kwargs.get('year')
	#get the directory one step up the tree from the filename to create a new subdir in the output dir 
	subdir=os.path.dirname(input_file).split('/')[-1] #this is hardcoded and should be changed to run on another platform 
	
	dir_name = os.path.join(output_dir,subdir)
	try: 
		if not os.path.exists(dir_name):
			print('making dir ',dir_name) 
			os.mkdir(dir_name)
		else: 
			pass
	except Exception as e: 
		print(f'That produced the {e} error')
	gdal.UseExceptions()
	
	#create the name of the output
	dst_layername = os.path.join(dir_name,f'{model}_model_{year}_{subdir}')
	if not os.path.exists(dst_layername+'.shp'): 
		#open a raster file and get some info 
		src_ds = gdal.Open(str(input_file))
		srs = osr.SpatialReference()
		srs.ImportFromWkt(src_ds.GetProjection())
		srcband = src_ds.GetRasterBand(1) #hard coded for a one band raster
		
		drv = ogr.GetDriverByName("ESRI Shapefile")
		dst_ds = drv.CreateDataSource(dst_layername + ".shp")
		dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs)
		newField = ogr.FieldDefn('rgi_label', ogr.OFTInteger)
		dst_layer.CreateField(newField)

		#write it out 
		gdal.Polygonize(srcband, None, dst_layer, 0, [], 
		callback=None )
		src_ds=None

		return dst_ds 
	else: 
		pass
```

Virtual rasters:  
```
def make_multiband_rasters(paths,output_dir,years,bands,dir_type,separate,noData_value,band_type): 
	'''
	Read in the list of tifs in a directory and convert to multiband vrt file.
	'''
	#for k,v in paths.items(): 
	outvrt = output_dir+f'{dir_type}_{years}_{band_type}_multiband.vrt'#f'optical_topo_multiband_{k}_tile_yearly_composite_{years[bands]}_w_class_label.vrt' #this is dependent on the dictionary of bands:years supplied below and assumes datatype Float32
	#print('the small tile output file is: ', outvrt)
	if not os.path.exists(outvrt): 
		if separate: 
			print('processing with separate')
			outds = gdal.BuildVRT(outvrt, paths, separate=True,bandList=[bands],srcNodata=noData_value)
		else: 
			print('processing without separate')
			outds = gdal.BuildVRT(outvrt, paths,bandList=[bands],srcNodata=noData_value)
	return None

```

### Numpy 

Reading rasters: 

```
def read_raster(self,raster):
	ds = gdal.Open(raster)
	band = ds.GetRasterBand(self.band)
	return band.ReadAsArray(),ds
```

Basic array math: 
```
#multiply the labels by a binary glacier map- this leaves 1 where CNN says glacier but RGI doesn't 
masked = mask*c_arr

#check how many pixels are missing labels 
num_label_pix = masked[masked==1].sum()
```

Masking and padding: 
```
def max_filtering(self): 
	"""Apply a max filter to numpy array. Designed to be applied in iteration."""

	#apply the max filter 
	labels1 = ndimage.maximum_filter(self.label_arr, size=3, mode='constant') #formerly labels 
	
	#get a mask which is just the pixels to be infilled 
	new_mask = labels1*self.base_arr #formerly ones 

	#fill in pixels that overlap between max arr and CNN class outside RGI labels 
	output = np.ma.array(self.input_arr,mask=new_mask).filled(new_mask) #formerly masked

	return output

def pad_width(self,small_arr,big_arr): 
	"""Helper function."""
	col_dif = big_arr.shape[1]-small_arr.shape[1] 
	
	return np.pad(small_arr,[(0,0),(0,col_dif)],mode='constant')

```

Boolean masking: 
```
#ones = np.where(masked>0,1,0)
ones = np.where(masked==1,1,0)
		
#get all the areas that have RGI ids 
labels = np.where(masked!=1,masked,0)
```

### Rasterio 

Resampling from [Rasterio](https://rasterio.readthedocs.io/en/latest/topics/resampling.html): 
```
import rasterio
from rasterio.enums import Resampling

upscale_factor = 2

with rasterio.open("example.tif") as dataset:

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )
```

Moving windows: 
```
def extract_chips(input_raster,filename,col_off,row_off,image_chip_size=128): 
	"""Read a vrt raster into np array."""

	subset_window=Window(col_off, row_off, image_chip_size, image_chip_size)
	
	with rasterio.open(input_raster) as src: #window works like Window(col_off, row_off, width, height)

		w = src.read(window=subset_window,masked=True)

		#get the image metadata for writing 
		profile=src.profile
		
		#make a few updates to the metadata before writing
		profile.update(
		driver='GTiff',
		compress='lzw', 
		width=image_chip_size, 
		height=image_chip_size,
		transform=rasterio.windows.transform(subset_window,src.transform)) #reassign the output origin to the subset_window 
		
	with rasterio.open(filename, 'w', **profile) as dst:
			dst.write(w)

	return None 
```

## Vectors

### Pandas 

Open csvs and make them into dataframes 
```
dfs = []
	for file in glob.glob(optical_dir+'*.csv'): 
		df = pd.read_csv(file)
		dfs.append(df)
	rs_data = pd.concat(dfs)
	rs_data = rs_data.dropna()
```

Format dates, columns and mask with booleans 
```
def leap_years(df,year_col='year'): 
	"""Determine whether a year is a leap year."""
	df['leap_year'] = np.where(df['year'] % 4 == 0, True, False)
	return df

def calc_climate_normal(df): 
	
	try: 
		df['year'] = df['date'].dt.year
		df['day'] = df['date'].dt.day
		df['month'] = df['date'].dt.month
	except KeyError: 
		print('Make sure the datetime col is called date')
	df['water_year'] = df.date.dt.year.where(df['date'].dt.month < 10, df['date'].dt.year + 1)
	#add day of water year (from 10/1)
	df['doy'] = df['date'].dt.dayofyear #df['water_year'] = df.datetime.dt.year.where(df.datetime.dt.month < 10, df.datetime.dt.year + 1)
	#adjust to the water year and decide if its a leap year 
	df = leap_years(df)
	#get instances where its not a leap year
	#df['dowy'] = np.where((df['month'] >= 1)  & (df['month']<10) & (df['leap_year'] == False),df['doy']+92,df['doy']-273)
	#then instances where it is not a leap year
	df['dowy'] = np.where((df['month'] >= 1)  & (df['month']<10), df['doy']+92,df['doy']-273)
	df['dowy'] = np.where((df['month'] >= 10) & (df['leap_year']==True),df['dowy']-1,df['dowy']) 
	#adjust for instances where it is a leap year 
	#df['dowy'] = np.where((df['month'] >= 1)  & (df['month']<10) & (df['leap_year']==True),df['dowy']-1,df['dowy'])
	# print('df here is: ')
	# print(df[['date','doy','dowy','leap_year','water_year']])
	return df
```

### Geopandas 

Read a shapefile, change or remove rows (features)
```
def remove_background_from_vector_file(input_file,field='rgi_label',crs='EPSG:3338',**kwargs): 
	"""Remove erroneous garbage from vector file which was converted from a raster."""
	if 'no_background' in str(input_file): 
		return None 
	else: 
		#create the output file name and check to see if it already exists
		output_file = str(input_file)[:-4]+'_no_background.shp'
		if not os.path.exists(output_file): 
			#read in a shapefile and sort the field that comes from conversion 
			try: 
				gdf = gpd.read_file(input_file).sort_values(field)
			except KeyError: 
				print(f'That shapefile does not have the field {field}.')
				raise
			#get only values above zero, this gets rid of the background garbage 
			gdf = gdf.loc[gdf[field]!=0] #changed 5/20/2021 so that it leaves neg values as those are the unlabled vectors we need to deal with in subsequent steps
			gdf['id'] = range(gdf.shape[0])
			gdf = gdf.to_crs(crs)
			print('The file to process is: ')
			print(output_file)
			#write the result to disk 
			try: 
				gdf.to_file(output_file)
			except ValueError: 
				print('That file does not appear to have any debris covered glacier.')
			return None 
		else: 
			pass
```

Edit attributes

```
def edit_and_filter_attributes(input_shp,crs='EPSG:3338',min_size=0.01,char_strip=-9): #the min size is given in km2 
	"""Used to create an area field in a shapefile attribute table and then remove items below a certain threshold."""
	print(input_shp)

	gdf = gpd.read_file(input_shp)
	try: 
		gdf = gdf.set_crs(crs)
	except AttributeError as e: 
		pass
		
	if 'r_area' not in gdf.columns: 
		gdf['r_area'] = gdf['geometry'].area / 10**6 #this assumes that the field that is being created is in meters and you want to change to km2

	gdf = gdf.loc[gdf['r_area']>=min_size]

	output_file = str(input_shp)[:char_strip]+f'_w_{min_size}_min_size.shp' #changed to just take off the '__buffer' and the ext
	try: 
		gdf.to_file(output_file)
	except ValueError: 
		print('there is no data in that file.')
	return output_file
```

### OGR

Buffer a shapefile: 

```
def createBuffer(input_file,output_dir,bufferDist=0,char_strip=-18,modifier=''):
	try: 
		output_file = os.path.join(output_dir,os.path.split(str(input_file))[1][:char_strip]+f'_buffer.shp')
		if not os.path.exists(output_file): 
			print('making ',str(input_file))
			inputds = ogr.Open(str(input_file))
			inputlyr = inputds.GetLayer()
			srs = osr.SpatialReference()
			srs.ImportFromEPSG(3338)
			shpdriver = ogr.GetDriverByName('ESRI Shapefile')
			outputBufferds = shpdriver.CreateDataSource(output_file)
			bufferlyr = outputBufferds.CreateLayer(output_file, srs, geom_type=ogr.wkbPolygon)
			featureDefn = bufferlyr.GetLayerDefn()

			#Create new fields in the output shp and get a list of field names for feature creation
			fieldNames = []
			for i in range(inputlyr.GetLayerDefn().GetFieldCount()):
				fieldDefn = inputlyr.GetLayerDefn().GetFieldDefn(i)
				bufferlyr.CreateField(fieldDefn)
				fieldNames.append(fieldDefn.name)

			for feature in inputlyr:
				ingeom = feature.GetGeometryRef()
				fieldVals = [] # make list of field values for feature
				for f in fieldNames: fieldVals.append(feature.GetField(f))
				geomBuffer = ingeom.Buffer(bufferDist)

				outFeature = ogr.Feature(featureDefn)
				outFeature.SetGeometry(geomBuffer)
				for v, val in enumerate(fieldVals): # Set output feature attributes
					outFeature.SetField(fieldNames[v], val)
				bufferlyr.CreateFeature(outFeature)
				outFeature = None
			inputds=None
		else: 
			print('That file already exists...')
	except Exception as e: 
		print('There was an error and it was: ', e)
		raise#print(f'Error was {e}')
```

## Other

### Matplotlib and Seaborn 

Lineplots with inset and custom legend 

![Example plot](https://drive.google.com/file/d/1UktkpsaX0DwEoVR-8krsiMV-NblM4VDU/view?usp=sharing)

```
def plot_side_by_side(df,output_dir): 

	black = Color("black")
	start_color = Color('darkblue')
	color_ls = list(start_color.range_to(Color("darkred"),19))
	color_ls = [c.hex_l for c in color_ls]
	colors_dict = dict(zip(range(1986,2022,2),color_ls))

	fig,(ax1,ax2) = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'wspace':0,'hspace':0})
	
	df['styles'] = np.where(df['year'] >= 1990,1,0)
	line_styles = ['-','--']
	styles_dict = {1: '', 0:(4,1.5)}
	cmap = plt.cm.RdGy_r
	N=18
	rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
	colors = plt.cm.RdGy_r(np.linspace(0.0,1.0,N)) # This returns RGBA; convert:
	
	#create a custom legend
	labels = sorted([str(y) for y in df['year'].unique()])
	#create a custom legend so we can get the dashed lines in there
	custom_lines = [Line2D([0], [0], c=colors[0][:-1], lw=2,label=labels[0],ls='--'),
	                Line2D([0], [0], c=colors[1][:-1], lw=2,label=labels[1],ls='--'),
	                Line2D([0], [0], c=colors[2][:-1], lw=2,label=labels[2]),
	                Line2D([0], [0], c=colors[3][:-1], lw=2,label=labels[3]),
	                Line2D([0], [0], c=colors[4][:-1], lw=2,label=labels[4]),
	                Line2D([0], [0], c=colors[5][:-1], lw=2,label=labels[5]),
	                Line2D([0], [0], c=colors[6][:-1], lw=2,label=labels[6]),
	                Line2D([0], [0], c=colors[7][:-1], lw=2,label=labels[7]),
	                Line2D([0], [0], c=colors[8][:-1], lw=2,label=labels[8]),
	                Line2D([0], [0], c=colors[9][:-1], lw=2,label=labels[9]),
	                Line2D([0], [0], c=colors[10][:-1], lw=2,label=labels[10]),
	                Line2D([0], [0], c=colors[11][:-1], lw=2,label=labels[11]),
	                Line2D([0], [0], c=colors[12][:-1], lw=2,label=labels[12]),
	                Line2D([0], [0], c=colors[13][:-1], lw=2,label=labels[13]),
	                Line2D([0], [0], c=colors[14][:-1], lw=2,label=labels[14]),
	                Line2D([0], [0], c=colors[15][:-1], lw=2,label=labels[15]),
	                Line2D([0], [0], c=colors[16][:-1], lw=2,label=labels[16]),
	                Line2D([0], [0], c=colors[17][:-1], lw=2,label=labels[17])
	                ]


	ax1.legend(handles=custom_lines, loc='upper right',ncol=2)

	#add a couple of bolder grid lines 
	ax2.plot(df['elev_band_max'].unique(),
		np.zeros(shape=len(df['elev_band_max'].unique())),
		linewidth=2,
		c='gray', 
		ls = '--', 
		alpha=0.5
		)

	#add some vertical elevation lines 
	for xi in range(0,5000,500): 
		ax1.axvline(x=xi, color='gray',ls='--',lw=1,alpha=0.25)
		ax2.axvline(x=xi, color='gray',ls='--',lw=1,alpha=0.25)

	sns.lineplot(x='elev_band_max', 
				 y='area',  
				 data=df,
				 hue='year', 
				 style='styles',
				 dashes=styles_dict,
				 palette=cmap,
				 linewidth=1.5,
				 ax=ax1, 
				 legend=False
				 )

	sns.lineplot(x='elev_band_max', 
				 y='mean_temp',  
				 data=df,
				 hue='year', 
				 palette=cmap,
				 linewidth=1.5,
				 ax=ax2, 
				 legend=False
				 )
	#handle the labels
	ax1.set_ylabel('Glacier area ($km^2$)',fontsize=10)
	ax2.set_ylabel('Mean annual \ntemp ($^\circ$C)',fontsize=10)
	ax2.set_xlabel('Elevation band max (m)',fontsize=10)
	ax1.grid(axis='both',alpha=0.25)
	ax2.grid(axis='both',alpha=0.25)
	ax1.set_xlim(200,5000)
	ax2.set_xlim(200,5000)

	# plt.show()
	# plt.close('all') 

	plot_fn = os.path.join(output_dir,'revised_200m_bands_temp_area_elev_plot_amended_lines.jpg')
	if not os.path.exists(plot_fn): 
		plt.savefig(plot_fn, 
					dpi=500,
					bbox_inches = 'tight',
					pad_inches = 0.1
					)

```

### Google Earth Engine Python API 

import ee

"""
Use the Google Earth Engine Python API to create tasks for generating yearly/seasonal Daymet estimates 
for HUCx river basins in the US Pacific Northwest (PNW). NOTE that you need to have an active GEE and associated 
Google Drive account for this script to work. You will be prompted to log into your account in the ee.Authenticate call. 

Inputs: 
Specify the HUC level 
Include any other inputs to spatially bound the river basins- here we use SNOTEL stations. 

Outputs: 
Script will start spatial statistics tasks (GEE command Export.table.toDrive) on your GEE account and export 
to the associated Google Drive account and specified folder. 
"""

#deal with authentication
try: 
	ee.Initialize()
except Exception as e: 
	ee.Authenticate()
	ee.Initialize()

class GetPrism(): 
	"""Sets up the Daymet imageCollection for GEE. Calculates an average 
	temperature band from tmin and tmax and adds that to the collection. 
	Performs spatial and temporal filtering. 
	"""

	def __init__(self,start_year,start_month,end_month,**kwargs): 
		self.start_year=start_year
		self.start_month=start_month
		self.end_month=end_month
		#set the rest of the vars from kwargs (end dates)

	def get_data(self): 
		prism_ic = (ee.ImageCollection("OREGONSTATE/PRISM/AN81d")#.filterBounds(self.aoi) #not 100% sure why this was filtering to the first aoi 9/6/2021
																.filter(ee.Filter.calendarRange(self.start_year,self.start_year,'year'))
																.filter(ee.Filter.calendarRange(self.start_month,self.end_month,'month'))
																.select(['ppt','tmean']))
															  #.filter(ee.Filter.calendarRange(self.start_day,self.end_day,'day_of_month')))
		return prism_ic

class ExportStats(): 

	def __init__(self,ic,features,scale=4000,**kwargs): #make sure to define a different start date if you want something else 
		self.ic=ic
		self.scale=scale
		self.features = features#kwargs.get('features')

		for key, value in kwargs.items():
			setattr(self, key, value)

	def calc_export_stats(self,feat,img): 
		"""Use a reducer to calc spatial stats."""
		# var get_ic_counts = comp_ic.map(function(img){ 
		pixel_ct_dict = img.reduceRegion(
			reducer=self.reducer, #maybe change to median? This get's the basin mean for a given day (image)
			geometry=feat.geometry(),
			scale=self.scale,
			crs='EPSG:4326', 
			tileScale=4,
			maxPixels=1e13
			)
		dict_out = pixel_ct_dict.set(self.huc,feat.get(self.huc)).set('date',ee.Date(img.get('system:time_start')))
		dict_feat = ee.Feature(None, dict_out)
		return dict_feat

	def generate_stats(self): 
		"""Iterator function for the calc_export_stats function. 
		This emulates the nested functions approach in GEE Javascript API."""

		get_ic_stats=ee.FeatureCollection(self.features).map(lambda feat: self.ic.map(lambda img: self.calc_export_stats(feat,img)))
		return get_ic_stats

	def run_exports(self): 
		"""Export some data."""

		task=ee.batch.Export.table.toDrive(
			collection=ee.FeatureCollection(self.generate_stats()).flatten(),
			description= f'proj_prism_mean_stats_for_{self.modifier}_{self.timeframe}', 
			folder=self.output_folder,
			fileNamePrefix=f'proj_prism_mean_stats_for_{self.modifier}_{self.timeframe}',
			fileFormat= 'csv'
			)
		#start the task in GEE 
		print(task)
		task.start()

def main(hucs): 

	for wy in range(1990,2021): #years: this is exclusive 
		
		for m in [[11,12],[1,2],[3,4]]: #these are the seasonal periods (months) used in analysis 	
			#generate the water year name first because that should be agnostic of the month 
			timeframe = f'start_month_{m[0]}_end_month_{m[1]}_WY{wy}'
			try: 
				if m[1] == 12: 
					#for the fall years, set the water year back one integer year. 
					#Note that this will still be labeled as the water year (preceeding year)
					amended_wy = wy-1
				else: 
					#for the winter and spring months reset the water year 
					#to the original value because we don't want the previous year for that one
					amended_wy = wy 
				ic = GetPrism(
					start_year=amended_wy,
					start_month=m[0],
					end_month=m[1]
					).get_data()
			except IndexError as e: 
				pass 
			
			#run the exports- note that default is to generate stats for a HUC level (e.g. 6,8) but this can be run as points (e.g. snotel). 
			#you should change the reducer to first and then make sure to change the huc variable to whatever the point dataset id col is. 
			exports = ExportStats(ic,features=hucs,
								timeframe=timeframe,
								huc='site_num',
								reducer=ee.Reducer.first(), #change to mean for running basins, first for points
								output_folder='prism_outputs', 
								modifier='SNOTEL'
								).run_exports()

if __name__ == '__main__':
	#note that the default setup for running this is to use HUCS (polygons) which demands the mean reducer. 
	#one should also be able to run this in point mode 
	pnw_snotel = ee.FeatureCollection("users/ak_glaciers/NWCC_high_resolution_coordinates_2019_hucs")
	hucs = ee.FeatureCollection("USGS/WBD/2017/HUC06").filterBounds(pnw_snotel)
	
	main(pnw_snotel)