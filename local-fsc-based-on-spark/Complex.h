/**
@file	Complex.h
@brief	Class for complex numbers
@author Bernard Heymann
@date	Created: 20050405  	    Modified: 20130618
**/

#include <cmath>
#include <iostream>
using namespace std;

#ifndef _Complex_
#define _Complex_
#undef Complex		// Complex is defined in X.h
/************************************************************************
@Object: class Complex
@Description:
	Template class for complex numbers.
@Features:
	The internal variables are two numbers, integer or floating point depending on the type.
*************************************************************************/
template <typename Type>
class Complex {
	Type	re, im;
public:
	Complex() : re(0),im(0) {}		// Constructors
	Complex(const Complex& c) : re(c.re),im(c.im) {}
	Complex(const Type r) : re(r),im(0) {}
	Complex(const Type r, const Type i) : re(r),im(i) {}
	Complex operator=(const double& d) {	// Operators
		re = d;
		im = 0;
		return *this;
	}
	Complex operator=(const Complex& c) {	// Operators
		re = c.re;
		im = c.im;
		return *this;
	}
	Complex operator-() {
		return Complex<Type>(-re, -im);
	}
	Complex operator+=(const Complex& c) {
		re += c.re;
		im += c.im;
		return *this;
	}
	Complex operator+(const Complex& c) {
		return Complex<Type>(re + c.re, im + c.im);
	}
	Complex operator-=(const Complex& c) {
		re -= c.re;
		im -= c.im;
		return *this;
	}
	Complex operator-(const Complex& c) {
		return Complex<Type>(re - c.re, im - c.im);
	}
	Complex operator*=(const double d) {
		re = Type(re*d);
		im = Type(im*d);
		return *this;
	}
	Complex operator*=(const Complex& c) {
		Complex<Type>	cn(re*c.re - im*c.im, re*c.im + im*c.re);
		*this = cn;
		return *this;
	}
	Complex operator*(const double d)	{
		return Complex<Type>(re*d, im*d);
	}
	Complex operator*(const Complex& c)	{
		return Complex<Type>(re*c.re - im*c.im, re*c.im + im*c.re);
	}
	Complex operator/=(const double d) {
		re = Type(re/d);
		im = Type(im/d);
		return *this;
	}
	Complex operator/=(const Complex& c) {
		Complex<Type>	cc(c);
		Type	d = (double)c.re*c.re + (double)c.im*c.im;
		cc.im = -cc.im;
		*this = (*this * cc)/d;
		return *this;
	}
	Complex operator/(const double d) {
		return Complex<Type>(re/d, im/d);
	}
	Complex operator/(const Complex& c)	{
		Complex<Type>	cc(c);
		double	d = (double)c.re*c.re + (double)c.im*c.im;
		cc.im = -cc.im;
		return (*this * cc)/d;
	}
	Type&	operator[](const int i) { return (i==0)? re: im; }
	template <typename T2> operator Complex<T2>() const {
		return Complex<T2>(re, im);
	}
	bool	is_finite() { return isfinite(re) & isfinite(im); }
	Type	real() { return re; }
	Type	imag() { return im; }
	double	power() { return (double)re*re + (double)im*im; }
	double	amp() { return sqrt((double)re*re + (double)im*im); }
	double	phi() { return atan2((double)im, (double)re); }
	Complex conj() { return Complex<Type>(re, -im); }
	void	set(const double a, const double p) { re = a*cos(p); im = a*sin(p); }
	void	real(const Type d) { re = d; }
	void	imag(const Type d) { im = d; }
	void	amp(const double d) { Type r = d/amp(); re *= r; im *= r; }
	void	phi(const double d) { Type a = amp(); re = a*cos(d); im = a*sin(d); }
	void	shift_phi(const double d) { Complex c((Type)cos(d), (Type)sin(d)); *this *= c; }
	Complex	unpack_first(Complex c) {
		return Complex<Type>(0.5L*(re + c.re), 0.5L*(im - c.im));
	}
	Complex	unpack_second(Complex c) {
		return Complex<Type>(0.5L*(c.im + im), 0.5L*(c.re - re));
	}
};

template <typename Type>
ostream& operator<<(ostream& output, Complex<Type>& c) {
	output.setf(ios::fixed, ios::floatfield);
	output.precision(4);
	output << "{" << c.real() << "," << c.imag() << "}";
	return output;
}

template <typename Type>
inline Complex<Type>	operator*(const double d, Complex<Type>& c)
{
	return c * d;
}

template <typename Type>
inline Complex<Type>	complex_unpack_combined_first(Complex<Type> c1, Complex<Type> c2)
{
	return (c1 + c2.conj()) * 0.5;
}

template <typename Type>
inline Complex<Type>	complex_unpack_combined_second(Complex<Type> c1, Complex<Type> c2)
{
	return Complex<Type>(0.5*(c2.imag() + c1.imag()), 0.5*(c2.real() - c1.real()));
}

#endif

