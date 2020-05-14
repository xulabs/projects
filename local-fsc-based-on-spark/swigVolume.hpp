/****************************************************************************//**
 * \file swigVolume.hpp
 * \brief The header file for the class swigTom::swigVolume.
 * \author  Thomas Hrabe
 * \version 0.2
 * \date    1.12.2008
 *******************************************************************************/

#ifndef SWIGVOLUME_HPP_
#define SWIGVOLUME_HPP_


#include <tom/volume.hpp>
#include <fftw3.h>
#include <tom/volume_fcn.hpp>
#include <tom/io/io.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "Complex.h" // cite bsoft's Complex

namespace swigTom{

union TypePointer {
        unsigned char*  uc;
        signed char*    sc;
        unsigned short* us;
        short*                  ss;
        unsigned int*   ui;
        int*                    si;
        unsigned long*  ul;
        long*                   sl;
        float*                  f;
        double*                 d;
};

	Complex<double> complex_new( const TypePointer tp, long j, long datasize_t2)
	{
    		long i2 = 2*j;
  		if (i2 >= datasize_t2) return 0;
    		return Complex<double>(tp.f[i2], tp.f[i2+1]);
	};

	double cut(double *data, long nr, long icol, double threshold, int dir=1)
        {
                long    i, i1, j, j1;
                double  val((data[0]+data[1])/2);
                for ( i=0, i1=1, j=icol*nr, j1=j+1; i1<nr; i++, i1++, j++, j1++ )
                {
                        if ( dir < 0 ) {  
                                if ( data[j] >= threshold && data[j1] < threshold )
                                        val = data[i] + (data[i1] - data[i])*(threshold - data[j])/(data[j1] - data[j]);
                        } else {   
                                if ( data[j] <= threshold && data[j1] > threshold )
                                        val = data[i] + (data[i1] - data[i])*(threshold - data[j])/(data[j1] - data[j]);
                        }
                }
                if ( val < data[1] ) {
                        if ( dir < 0 ) {
                                if ( data[j] > threshold ) val = data[i];
                        } else {
                                if ( data[j] < threshold ) val = data[i];
                        }
                }
                return val;
        };
	//
	void set_d(float *v3d_f, int j, double v, long datasize_t)
	{
   		if ( j >= datasize_t ) return;
   		v3d_f[j] = v;
   		return;
   	}


	unsigned char * padding(unsigned char *duc, long pad, int sizex, int sizey, int sizez, int fill_type, double fill)
        {
                long x,y,z;
                x = y = z = (long)sizex;

                int  c=1;
                long nusize[3] = {2*x, 2*y, 2*z};
                long translate[3] = {0, 0, 0};
                long i, j, xx, yy, zz;
                long oldx, oldy, oldz;
		long elementsize = c*sizeof(float);
                long nualloc = (long) nusize[0]*nusize[1]*nusize[2]*elementsize;

                unsigned char *nudata = new unsigned char[nualloc];
                TypePointer fp;
                fp.uc  = new unsigned char[sizeof(float)];
                fp.f[0] = fill;
                

		for (zz=0, i=0; zz<nusize[2]; zz++){
                        oldz = (long)zz - translate[2];
                        for (yy=0; yy<nusize[1]; yy++){
                                oldy = (long)yy - translate[1];
                                for (xx=0; xx<nusize[0]; xx++, i++){
                                        oldx = (long)xx - translate[0];
                                        if (oldx < 0  || oldx >= x || oldy < 0 ||
                                            oldy >= y || oldz < 0  || oldz >= z){
                                                memcpy(nudata+i*elementsize, fp.uc, elementsize);
                                        } else {
                                                j = (oldz*y + oldy)*x + oldx;
						memcpy(nudata+i*elementsize, duc+j*elementsize, elementsize);
                                        }
                                }
                        }
                }

                delete[] fp.uc;
                fp.uc = NULL;

                return nudata;                          
	};


/**********************************************************************//**
 *	\brief Volume class used for wraping tom::Volume
 *
 *
 *	swigVolume wraps tom::Volume<T> for interfacing with SWIG.
 * 	Methods defined here will be visible in the interfaced language such as Python.
 *	There will not be much of a documentation for class methods because they basically wrap baseclass methods.
 *	Template parameters have become complicated because of the predefined template functions / members of the parent classes.
 */
template<typename T, typename TSCALE_SHIFT>
class swigVolume : public tom::Volume<T>{

private:
	float ftSizeX;
	float ftSizeY;
	float ftSizeZ;

public:

	//swigVolume(std::size_t sizex,std::size_t sizey,std::size_t sizez): tom::Volume<T>(sizex,sizey,sizez,&fftw_malloc,&fftw_free){ this->ftSizeX = 0;this->ftSizeY = 0;this->ftSizeZ = 0; };
	swigVolume(std::size_t sizex,std::size_t sizey,std::size_t sizez): tom::Volume<T>(sizex,sizey,sizez,NULL,NULL){ this->ftSizeX = 0;this->ftSizeY = 0;this->ftSizeZ = 0; };
	swigVolume(const tom::Volume<T>& v) : tom::Volume<T>(v){ this->ftSizeX = 0;this->ftSizeY = 0;this->ftSizeZ = 0; };
	swigVolume(const swigVolume<T,TSCALE_SHIFT>& v) : tom::Volume<T>(v){ this->ftSizeX = v.getFtSizeX();this->ftSizeY = v.getFtSizeY();this->ftSizeZ = v.getFtSizeZ(); };
	
	swigVolume(T *data, std::size_t sizex, std::size_t sizey, std::size_t sizez, std::size_t stridex, std::size_t stridey, std::size_t stridez) : tom::Volume<T>(data, sizex, sizey, sizez, stridex, stridey, stridez, false, NULL)
	{
		this->ftSizeX = 0;this->ftSizeY = 0;this->ftSizeZ = 0;
	};

	~swigVolume(){};


	/**
	 *	\brief Wrapper function
	 */
	std::size_t sizeX() const{
		return this->getSizeX();
	};
	/**
	 *	\brief Wrapper function
	 */
	std::size_t sizeY() const{
		return this->getSizeY();
	};
	/**
	 *	\brief Wrapper function
	 */
	std::size_t sizeZ() const{
		return this->getSizeZ();
	};

	float getFtSizeX() const{
		return this->ftSizeX;
	};
	/**
	 *	\brief Wrapper function
	 */
	float getFtSizeY() const{
		return this->ftSizeY;
	};
	/**
	 *	\brief Wrapper function
	 */
	float getFtSizeZ() const{
		return this->ftSizeZ;
	};

	void setFtSizeX(float sizeX){
		this->ftSizeX=sizeX;
	};
	/**
	*	\brief Wrapper function
	*/
	void setFtSizeY(float sizeY){
		this->ftSizeY=sizeY;
	};
	/**
	*	\brief Wrapper function
	*/
	void setFtSizeZ(float sizeZ){
		this->ftSizeZ=sizeZ;
	};
	/**
	 *	\brief Wrapper function
	 */

	std::size_t strideX() const{
		return this->getStrideX();
	};
	/**
	 *	\brief Wrapper function
	 */
	std::size_t strideY() const{
		return this->getStrideY();
	};
	/**
	 *	\brief Wrapper function
	 */
	std::size_t strideZ() const{
		return this->getStrideZ();
	};
	/**
	 *	\brief Wrapper function
	 */
	void write(std::string fileName){
		tom::io::write_to_em(*this,fileName,NULL);
	};

	void write(std::string fileName,std::string fileType){

		if(fileType == "mrc")
			tom::io::write_to_mrc(*this,fileName,NULL);
		else if(fileType == "ccp4")
			tom::io::write_to_ccp4(*this,fileName,NULL);
		else if(fileType == "em")
			tom::io::write_to_em(*this,fileName,NULL);
		else{
			std::cout << std::endl << "Filetype unknown, saving data as EM to " << fileName << std::endl;
			tom::io::write_to_em(*this,fileName,NULL);
		}
	};


	/**
	 *	\brief Wrapper function
	 */
	void info (const std::string &name) const{
		this->printInfo(name);
	}
	/**
	 *	\brief Wrapper function
	 */
	std::size_t numelem() const{
		return this->numel();
	}
	/**
	 *	\brief Wrapper function
	 */
	bool equalsTo(const swigTom::swigVolume<T,TSCALE_SHIFT> &v) const{
		return (*this) == v;
	}
	/**
	 *	\brief Wrapper function
	 */
	T getV(std::size_t x,std::size_t y, std::size_t z){
		return this->get(x,y,z);
	}
	/**
	 *	\brief Wrapper function
	 */
	void setV(T val,std::size_t x,std::size_t y, std::size_t z){
		this->get(x,y,z) = val;
	}
	/**
	 *	\brief Wrapper function
	 */
	void setAll(T val){
		this->setValues(val);
	}
	/**
	 *	\brief Wrapper function
	 */
	void copyVolume(const swigTom::swigVolume<T,TSCALE_SHIFT>& v2){
		this->setValues(v2);
		this->ftSizeX = v2.getFtSizeX();
		this->ftSizeY = v2.getFtSizeY();
		this->ftSizeZ = v2.getFtSizeZ();
	}
	/**
	 *	\brief Wrapper function
	 */
	void shiftscale(const TSCALE_SHIFT & shiftV, const TSCALE_SHIFT & scaleV){
		this->shift_scale(shiftV,scaleV);
	}

	/**
	 *	\brief Wrapper function
	 */
	std::size_t dims(){
		if(this->getSizeZ() >0)
			return 3;
		else if(this->getSizeY() >0)
			return 2;
		else
			return 1;
	}

	/*
	Complex<double> complex_new( const TypePointer tp, long j)
	{
    		long i2 = 2*j;
    		//return Complex<double>((*this).d[i2], (*this).d[i2+1]);
    		return Complex<double>(tp.f[i2], tp.f[i2+1]);
	};

	double cut(double *data, long icol, double threshold, int dir=1);

	//unsigned char * padding(const swigVolume<T,T> volume, long pad, int fill_type, double fill);
	unsigned char * padding(unsigned char *duc, long pad, int sizex, int sizey, int sizez, int fill_type, double fill);
	*/

	swigVolume<T,TSCALE_SHIFT> operator+(const swigVolume<T,TSCALE_SHIFT> &otherVol) const;
	swigVolume<T,TSCALE_SHIFT> operator+(const TSCALE_SHIFT &value) const;
	swigVolume<T,TSCALE_SHIFT> operator*(const swigVolume<T,TSCALE_SHIFT> &otherVol) const;
	swigVolume<T,TSCALE_SHIFT> operator*(const TSCALE_SHIFT &value) const;
	swigVolume<T,TSCALE_SHIFT> operator-(const swigVolume<T,TSCALE_SHIFT> &otherVol) const;
	swigVolume<T,TSCALE_SHIFT> operator-(const TSCALE_SHIFT &value) const;
	swigVolume<T,TSCALE_SHIFT> operator/(const swigVolume<T,TSCALE_SHIFT> &otherVol) const;
	swigVolume<T,TSCALE_SHIFT> operator/(const TSCALE_SHIFT &value) const;
	const T operator()(const std::size_t &x,const std::size_t &y,const std::size_t &z);
	void operator()(const T &value,const std::size_t &x,const std::size_t &y,const std::size_t &z);
};


// The following is the class for EM header
/*
class EMHeader {

	tom_io_em_header raw_data;

public:

	EMHeader() {
		this->raw_data.machine = 6; // set the machine code to PC
	};
	
	~EMHeader() {
		delete raw_data;
	};
	
	void set_voltage(long voltage)
	{
		this->raw_data.emdata[0] = voltage;
	}
	
};
*/


}


#endif
