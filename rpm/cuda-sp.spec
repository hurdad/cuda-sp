Name:           cuda-sp
Version:	%{VERSION}
Release:        1%{?dist}
Summary:        Cuda Spectral Periodgram Library
License:        Apache2
Group:          Development/Libraries/C and C++
Url:            https://github.com/hurdad/cuda-sp
Source:         %{name}-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
BuildRequires:  make
BuildRequires:  gcc-c++

%description
Cuda Spectral Periodgram Library

%package devel
Summary:    Development headers and library for %{name}
Group:      Development/Libraries
Requires:   %{name}%{?_isa} = %{version}-%{release}

%description devel
This package contains the development headers and library for %{name}.

%prep
%setup -n %{name}-%{version}

%build
make %{?_smp_mflags}

%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

%clean
rm -rf $RPM_BUILD_ROOT

%post
ldconfig

%postun
ldconfig

%files
%defattr(-,root,root,-)
%doc LICENSE README.md
%{_libdir}/lib%{name}.so

%files devel
%defattr(-,root,root,-)

%{_includedir}/%{name}/

%changelog

