
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    h, w = img.shape[:2]
    hh = np.int(np.ceil(h / 2))
    hw = np.int(np.ceil(w / 2))
    mask_str = np.tile(np.array(list(pattern)).reshape(2, 2), [hh, hw])[:h, :w]
    mask = np.zeros((h, w), dtype=np.float64)
    mask[mask_str == 'G'] = 1
    mask[mask_str == 'R'] = fr
    mask[mask_str == 'B'] = fb
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask