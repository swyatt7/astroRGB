import glob
import sep
import numpy as np
from astropy.wcs import WCS
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as interp

#real dirty way to find offsets to stack
def find_offset(ref_x, ref_y, _data):
    #Get the objects in our current image
    _data = _data.byteswap().newbyteorder()
    _data_bkg = sep.Background(_data)
    data_objs = sep.extract(_data-_data_bkg, thresh=10.0, err=_data_bkg.globalrms, minarea=20)
    
    #loop over them and calculate their distances
    for i,j in enumerate(data_objs['x']):
        shift_x = ref_x - data_objs['x'][i]
        shift_y = ref_y - data_objs['y'][i]
        distance = np.sqrt((shift_x)**2+(shift_y)**2)
        
        # if the distance is less than this threshhold, its a match
        # return the offset x,y
        if distance < 10:
            return shift_x, shift_y
            break
    return None, None

class astroRGB():
    def __init__(self, filters=['Harris-R', 'Harris-V', 'Bessell-U'], fits_files=None, savefig=False, pltname=None):
        self.filters=filters
        self.fits_files = fits_files
        self.grouped_fits_files = {}
        self.savefig = savefig
        self.pltname = pltname

        #if no fits files are passed in
        # grab the ones in the current directory
        if self.fits_files is None:
            self.fits_files = files = glob.glob('*fits')
            if not len(self.fits_files):
                assert False, 'No fits files found to create RBG image'

        if not len(self.filters):
            assert False, "No filters...?"

        #create the grouped_fits_files object with the passed in filters as a key
        for _filter in self.filters:
            self.grouped_fits_files[_filter] = []

        available_filters = []
        #create the grouped fits dictionary based off filter
        for f in [x for x in self.fits_files if 'samastro_stack' not in x]:
            _filter = pyfits.getheader(f)['FILTER']
            available_filters.append(_filter)
            if _filter in self.filters:
                self.grouped_fits_files[_filter].append(f)

        available_filters = list(set(available_filters))

        #if no fits files were found for a requested filter, alert
        for _filter in self.filters:
            if not len(self.grouped_fits_files[_filter]):
                print()
                assert False, 'No fits files corresponding to filter: {}.. Available filters in fits_files list: {}'.format(_filter, available_filters)

        if self.savefig and not self.pltname:
            self.pltname = 'ImaNinnyNoPlotName.png'        


    def stackimages(self, filters=[], offset_thresh=5, skipifexist=True):
        if filters is None or not len(filters):
            filters = self.filters

        ref_file = self.grouped_fits_files[list(self.grouped_fits_files.keys())[0]][0]
        ref_img = pyfits.getdata(ref_file)
        ref_img = ref_img.byteswap().newbyteorder()
        ref_bkg = sep.Background(ref_img)
        ref_objects = sep.extract(ref_img-ref_bkg, thresh=10.0, err=ref_bkg.globalrms, minarea=20)

        image_size = np.shape(ref_img)
        size = list(image_size)[0]

        for band in filters:
            print('Stacking all images in filter: {}'.format(band))
            outfname = 'samastro_stack_{}.fits'.format(band)
            allfiles = glob.glob('*.fits')
            
            if outfname not in allfiles or not skipifexist:
                _shift_data = np.zeros(image_size)
                
                #loop over the files associated with each band
                for file in self.grouped_fits_files[band]:
                    
                    #get the image data
                    tmp_data = pyfits.getdata(file)
                    
                    #our lists to hold the offsets
                    sx, sy = [], []
                    
                    #loop over each reference object
                    for i,j in enumerate(ref_objects['x']):
                        #find and append the offsets to our lists
                        tmp_sx, tmp_sy = find_offset(ref_objects['x'][i], ref_objects['y'][i], tmp_data)
                        if tmp_sx != None:
                            sx.append(tmp_sx)
                            sy.append(tmp_sy)

                    #calculate the average offset
                    if len(sx) and len(sy):
                        shift_x, shift_y = np.mean(sx), np.mean(sy)
                    else:
                        shift_x, shift_y = 0, 0

                    print('Average offsets in x: {}, y: {}\n'.format(round(shift_x, 3), round(shift_y, 3)))
                    if shift_x != 0 and shift_y != 0 and abs(shift_x) < offset_thresh and abs(shift_y) < offset_thresh:
                        #scipy method that shifts the image based off these offsets
                        new_data = interp.shift(tmp_data, [shift_y, shift_x])
                        #new coadded data from the shift
                        _shift_data += new_data
                try:
                    stacked_average_data = _shift_data/float(len(self.grouped_fits_files[band]))
                    hdu = pyfits.PrimaryHDU(stacked_average_data)
                    hdu.writeto(outfname, overwrite=True)
                except:
                    #this should only happen if no images were able to be stacked for a given filter
                    print('Fits images for filter {} could not be stacked. Make sure they are all of the same field'.format(band))
            else:
                print('Fits images for filter {} have already been stacked: {}'.format(band, outfname))
        

    def plot(self, rgb_scale=np.array([0.23, 0.22, 0.28])*1, quad=0.9, show=True, savefig=True):
        shift_band_image = {}
        ref_file = self.grouped_fits_files[list(self.grouped_fits_files.keys())[0]][0]
        ref_img = pyfits.getdata(ref_file)
        ref_img = ref_img.byteswap().newbyteorder()

        image_size = np.shape(ref_img)
        size = list(image_size)[0]
        simpleRGB=np.zeros((size,size,3),dtype=float)
        allfiles = glob.glob('*.fits')

        for band in self.filters:
            stacked_file = 'samastro_stack_{}.fits'.format(band)
            if stacked_file not in allfiles:
                self.stackimages(filters=[band])
            
            stacked_data = pyfits.getdata(stacked_file)

            shift_band_image.update({band: stacked_data})

        for i in range(len(self.filters)):
            data = shift_band_image[self.filters[i]].copy()
            min_value = np.quantile(data, [0.02, 1-0.02])[0]
            max_value = np.quantile(data, [0.02, 1-0.02])[1]
            data = (data - min_value)/(max_value-min_value)
            simpleRGB[:,:,i]=(data*rgb_scale[i])**quad
        
        ax = plt.subplot()
        ax.tick_params(which = 'both', size = 0, labelsize = 0)
        ax.imshow(simpleRGB, origin='lower')

        if savefig:
            plt.savefig(self.pltname, format='png', dpi=1000)

        if show:
            plt.show()

img = astroRGB(filters=['Harris-R', 'Harris-V', 'Bessell-U'], savefig=True)
img.stackimages(skipifexist=True)
img.plot(show=True, rgb_scale=np.array([0.23, 0.22, 0.28])*1, quad=0.9)