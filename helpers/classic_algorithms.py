from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import curve_fit
from scipy import stats
import numpy as np

class SecondDifference(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, channel=142, fwhm=3, threshold=-1.47, tol=1, pdf=False):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.threshold = threshold
        self.p = self.fwhm/2.355
        self.tol = tol
        self.pdf = pdf
        
    def fit(self, X=None, y=None):
        """
        A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier

        return self
    
    def _seconddiff(self, x, y=None):        
        c = [-100]
        j = 0
        new_c = lambda p, j: 100*(j**2 - p**2)/p**2 * np.exp(-0.5 * j**2/p**2) 
        for iteration in range(50):
            j += 1         
            c.append(new_c(self.p,j))
            if abs(new_c(self.p,j+1)) < 1:
                break
        c = np.array(c)
        c[1] = -np.sum(c) + c[1]
        c[1:] /= 2.0      
        c = np.hstack((c[::-1], c[1:]))
        #print c.sum()
        sd = np.sqrt(np.convolve(x, c**2, mode='same'))
        dd = np.convolve(x, c, mode='same')
        ss = dd/sd
        #plt.plot(c)
        return ss

    def predict(self, X, y=None):
        prediction = []        
        for x in X:
            transformed = self._seconddiff(x)
            peaklist = np.argwhere(transformed <= self.threshold)
            distances = abs(self.channel - peaklist)
            #print distances
            if (len(distances) > 0):
                if np.min(distances) <= self.tol:
                    prediction.append(1.0)
                else:
                    prediction.append(0.0)
            else:
                prediction.append(0.0)
        return np.array(prediction)
    
    def _prob(self, sig):
        if self.pdf:
            return stats.norm.cdf(sig**2, self.threshold**2, -self.threshold)
        else:
            return -sig
    
    def predict_proba(self, X, y=None):
        scores_pos = []        
        for x in X:
            # espectro transformado
            transformed = self._seconddiff(x)
            # canais onde ss é menor que threshold
            peaklist = np.argwhere(transformed <= self.threshold)
            # distância entre os canais onde ss é menor que threshold e o canal onde devia estar o pico
            distances = abs(self.channel - peaklist)
            # ss dos picos cuja distância do pico alvo eh menor que tol
            peaks_in_tol = transformed[peaklist[distances <= self.tol]]            
            # média das ss dos picos cuja distância do pico alvo é menor que tol
            if len(peaks_in_tol) > 0:
                mean_ss = np.mean(peaks_in_tol)
            else:
                mean_ss = transformed[self.channel]
            scores_pos.append(self._prob(mean_ss))            
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X) == y)/float(len(y)))
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self


class Manual(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""
    def __init__(self, channel=142, fwhm=3, alpha=0.05, l=4, fwhm_multiplier=1.25):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.alpha = alpha
        self.l = l
        self.fwhm_multiplier = fwhm_multiplier
        
    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """      
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier

        return self
    
    def _prob(self, x):
        m = int(round(self.fwhm*self.fwhm_multiplier, 0))        
        b = 2*m+1
        N1 = np.sum(x[self.channel-m-self.l:self.channel-m])
        N2 = np.sum(x[self.channel+m+1:self.channel+m+self.l+1])
        Ns = np.sum(x[self.channel-m:self.channel+m+1])            
        N0 = (N1 + N2)*b/(2.0*self.l)
        Nn = Ns - N0
        sigma_Nn = np.sqrt(Ns + N0*b/(2*self.l))
        prob = stats.norm.cdf(Nn, 0, sigma_Nn)
        return prob
    
    def predict_proba(self, X, y=None):        
        scores_pos = [self._prob(x) for x in X]        
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores        

    def predict(self, X, y=None):
        prediction = [self._prob(x) > (1-self.alpha) for x in X]
        return np.array(prediction).astype('float')   

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))   
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self


class LibCorNID(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""
    def __init__(self, channel=142, fwhm=3, sensitivity=0.53, tol=1):
        """
        Called when initializing the classifier
        """
        self.channel = channel
        self.fwhm = fwhm
        self.sensitivity = sensitivity
        self.tol = tol
        
    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """      
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier

        return self
    
    def _erosion(self, x, iterations=9):
        for n in range(1,iterations):
            m = int(round(3.0/n*self.fwhm,0))
            mask = np.zeros(2*m + 1)
            mask[0] = 0.5
            mask[-1] = 0.5        
            avg = np.convolve(x, mask, mode='same')            
            x = np.minimum(x,avg+2*np.sqrt(avg))
        return x
    
    def _gauss(self, x,a,x0):
        return a*np.exp(-(x-x0)**2/(2*(self.fwhm/2.355)**2))
    
    def _corr(self, x):
        transformed = x - self._erosion(x)        
        m = int(round(1.5*self.fwhm, 0))
        window = np.arange(self.channel-m, self.channel+m+1)
        count = x[self.channel-m:self.channel+m+1]          
        try:
            popt,_ = curve_fit(self._gauss,window,count,p0=[np.sum(count), self.channel])
            dist = abs(popt[0] - self.channel)
            #corr = abs(np.corrcoef(count, self._gauss(window, *popt)))[0][1]
            corr = np.corrcoef(count, self._gauss(window, *popt))[0][1]
            corr = 0 if corr < 0 else corr
        except RuntimeError:
            corr = 0       
        return corr
        

    def predict(self, X, y=None):
        prediction = [self._corr(x) > self.sensitivity for x in X]
        return np.array(prediction).astype('float')
    
    def predict_proba(self, X, y=None):
        scores_pos = [self._corr(x) for x in X]        
        scores_pos = np.array(scores_pos)
        scores_neg = 1 - scores_pos
        scores = np.hstack((scores_neg.reshape(-1,1), scores_pos.reshape(-1,1) ))
        return scores

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))
    
    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self