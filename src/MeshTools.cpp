#include <dolfin.h>

using namespace dolfin;

//b--c
//|  |
//a--d
//b--*    *--*
//|  | or |  |  two ends needed
//a--*    a--d
class RectBorderLine: public dolfin::SubDomain {
public:
	dolfin::Point p1;
	dolfin::Point p2;
	bool xdir;
	bool p1p2;
	RectBorderLine(dolfin::Point a, dolfin::Point b) {
		p1 = a;
		p2 = b;
		if(p1.x()==p2.x()) {//y-axis direction
			if(p1.y() <= p2.y())
				p1p2 = true;
			else
				p1p2 = false;
			xdir = false;
		}
		else if(p1.y()==p2.y()) {//x-axis direction
			if(p1.x() <= p2.x())
				p1p2 = true;
			else
				p1p2 = false;
			xdir = true;
		}
		else {
			std::cout<<"The points do not define a rectangle's border line"<<std::endl;
		}
	}
	bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
		if(xdir) {//x-axis direction
			if(p1p2) {
				return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (x[1] == p1.y());
			}
			else {
				return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (x[1] == p1.y());
			}
		}
		else{//y-axis direction
			if(p1p2) {
				return (p1.y() < x[1]) && (x[1] < p2.y()) && (x[0] == p1.x());
			}
			else {
				return (p1.y() > x[1]) && (x[1] > p2.y()) && (x[0] == p1.x());
			}
		}
	}
};

//		f--g
//roof side     |  |
//		e--h
//b--c
//|  |  floor side
//a--d
//*--c    *--f
//|  | or |  |  two ends needed
//a--*    a--*
//for compatibility no edge included in side planes
class BoxBorderPlane: public dolfin::SubDomain {
public:
	dolfin::Point p1;
	dolfin::Point p2;
	bool hrz;
	bool xdir;
	bool p1p2[2];
	BoxBorderPlane(dolfin::Point a, dolfin::Point b) {
		p1 = a;
		p2 = b;
		if(p1.z()==p2.z()) {//horizontal plane
			if(p1.x() <= p2.x())
				p1p2[0] = true;
			else
				p1p2[0] = false;
			if(p1.y() <= p2.y())
				p1p2[1] = true;
			else
				p1p2[1] = false;
			hrz = true;
		}
		else {//side plane
			if(p1.x()==p2.x()) {//zy plane
				if(p1.y() <= p2.y())
					p1p2[0] = true;
				else
					p1p2[0] = false;
				xdir = false;
			}
			else if(p1.y()==p2.y()) {//zx plane
				if(p1.x() <= p2.x())
					p1p2[0] = true;
				else
					p1p2[0] = false;
				xdir = true;
			}
			else
				std::cout<<"The points do not define a Box's border plane"<<std::endl;
			if(p1.z() <= p2.z())
				p1p2[1] = true;
			else
				p1p2[1] = false;
			hrz = false;
		}
	}
	bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
		if(hrz) {//horizontal plane
			if(p1p2[0]) {
				if(p1p2[1]) {
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (x[2] == p1.z());
				}
				else
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (x[2] == p1.z());
			}
			else {
				if(p1p2[1])
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (x[2] == p1.z());
				else
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (x[2] == p1.z());
			}
		}
		else {//side plane
			if(xdir) {//zx plane
				if(p1p2[0]) {
					if(p1p2[1]) {
						return (p1.x() < x[0]) && (x[0] < p2.x()) && (p1.z() < x[2]) && (x[2] < p2.z()) && (x[1] == p1.y());
					}
					else
						return (p1.x() < x[0]) && (x[0] < p2.x()) && (p1.z() > x[2]) && (x[2] > p2.z()) && (x[1] == p1.y());
				}
				else {
					if(p1p2[1])
						return (p1.x() > x[0]) && (x[0] > p2.x()) && (p1.z() < x[2]) && (x[2] < p2.z()) && (x[1] == p1.y());
					else
						return (p1.x() > x[0]) && (x[0] > p2.x()) && (p1.z() > x[2]) && (x[2] > p2.z()) && (x[1] == p1.y());
				}
			}
			else {//zy plane
				if(p1p2[0]) {
					if(p1p2[1]) {
						return (p1.y() < x[1]) && (x[1] < p2.y()) && (p1.z() < x[2]) && (x[2] < p2.z()) && (x[0] == p1.x());
					}
					else
						return (p1.y() < x[1]) && (x[1] < p2.y()) && (p1.z() > x[2]) && (x[2] > p2.z()) && (x[0] == p1.x());
				}
				else {
					if(p1p2[1])
						return (p1.y() > x[1]) && (x[1] > p2.y()) && (p1.z() < x[2]) && (x[2] < p2.z()) && (x[0] == p1.x());
					else
						return (p1.y() > x[1]) && (x[1] > p2.y()) && (p1.z() > x[2]) && (x[2] > p2.z()) && (x[0] == p1.x());
				}
			}
		}
	}
};

//simply consider the space (2D) between two points
class TwoPointsSpace2D: public dolfin::SubDomain {
public:
	dolfin::Point p1;
	dolfin::Point p2;
	bool p1p2[2];
	TwoPointsSpace2D(dolfin::Point a, dolfin::Point b) {
		p1 = a;
		p2 = b;
		if(p1.x() <= p2.x())
			p1p2[0] = true;
		else
			p1p2[0] = false;
		if(p1.y() <= p2.y())
			p1p2[1] = true;
		else
			p1p2[1] = false;
	}
	bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
		if(p1p2[1]) {
			if(p1p2[0]) {
				return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y());
			}
			else
				return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y());
		}
		else {
			if(p1p2[0])
				return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y());
			else
				return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y());
		}
	}
};


//simply consider the space (3D) between two points
class TwoPointsSpace3D: public dolfin::SubDomain {
public:
	dolfin::Point p1;
	dolfin::Point p2;
	bool p1p2[3];
	TwoPointsSpace3D(dolfin::Point a, dolfin::Point b) {
		p1 = a;
		p2 = b;
		if(p1.x() <= p2.x())
			p1p2[0] = true;
		else
			p1p2[0] = false;
		if(p1.y() <= p2.y())
			p1p2[1] = true;
		else
			p1p2[1] = false;
		if(p1.z() <= p2.z())
			p1p2[2] = true;
		else
			p1p2[2] = false;
	}
	bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
		if(p1p2[2]) {
			if(p1p2[1]) {
				if(p1p2[0]) {
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (p1.z() <= x[2]) && (x[2] <= p2.z());
				}
				else
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (p1.z() <= x[2]) && (x[2] <= p2.z());
			}
			else {
				if(p1p2[0])
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (p1.z() <= x[2]) && (x[2] <= p2.z());
				else
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (p1.z() <= x[2]) && (x[2] <= p2.z());
			}
		}
		else {
			if(p1p2[1]) {
				if(p1p2[0])
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (p1.z() >= x[2]) && (x[2] >= p2.z());
				else
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() <= x[1]) && (x[1] <= p2.y()) && (p1.z() >= x[2]) && (x[2] >= p2.z());
			}
			else {
				if(p1p2[0])
					return (p1.x() <= x[0]) && (x[0] <= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (p1.z() >= x[2]) && (x[2] >= p2.z());
				else
					return (p1.x() >= x[0]) && (x[0] >= p2.x()) && (p1.y() >= x[1]) && (x[1] >= p2.y()) && (p1.z() >= x[2]) && (x[2] >= p2.z());
			}
		}
	}
};

//simply consider the distance (2D) from the center
class CircularDomain: public dolfin::SubDomain {
public:
	dolfin::Point c;
	double r;
	CircularDomain(dolfin::Point a, double d) {
		c = a;
		r = d;
	}
	bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
		return (std::sqrt((x[0]-c.x())*(x[0]-c.x()) + (x[1]-c.y())*(x[1]-c.y())) < r);		
	}
};

int RectMeshGenerator(dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength, std::string DiagonalDirection) {
	double nx = std::floor((b.x() - a.x()) / ElementLength) + 2; //Number of elements (intervals)
	double ny = std::floor((b.y() - a.y()) / ElementLength) + 2;
	mesh = Mesh(RectangleMesh(a, b, nx, ny, DiagonalDirection)); //Rectangular mesh
	return 0;
}

int RectMeshGenerator(dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength) {
	double nx = std::floor((b.x() - a.x()) / ElementLength) + 2; //Number of elements (intervals)
	double ny = std::floor((b.y() - a.y()) / ElementLength) + 2;
	mesh = Mesh(RectangleMesh(a, b, nx, ny, "right")); //Rectangular mesh
	return 0;
}

int RectMeshGenerator(dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b) {
	mesh = Mesh(RectangleMesh(a, b, 100, 100, "right")); //Rectangular mesh
	return 0;
}

int BoxMeshGenerator(dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength) {
	auto nx = static_cast<unsigned long int>(std::floor((b.x() - a.x()) / ElementLength) + 2); //Number of elements (intervals)
	auto ny = static_cast<unsigned long int>(std::floor((b.y() - a.y()) / ElementLength) + 2);
	auto nz = static_cast<unsigned long int>(std::floor((b.z() - a.z()) / ElementLength) + 2);
	mesh = Mesh(BoxMesh(a, b, nx, ny, nz)); //Box mesh
	return 0;
}

int BoxMeshGenerator(dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b) {
	mesh = Mesh(BoxMesh(a, b, 100, 100, 100)); //Box mesh
	return 0;
}

int RectMeshGenerator(MPI_Comm comm, dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength, std::string DiagonalDirection) {
	double nx = std::floor((b.x() - a.x()) / ElementLength) + 2; //Number of elements (intervals)
	double ny = std::floor((b.y() - a.y()) / ElementLength) + 2;
	mesh = Mesh(RectangleMesh(comm, a, b, nx, ny, DiagonalDirection)); //Rectangular mesh
	return 0;
}

int RectMeshGenerator(MPI_Comm comm, dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength) {
	double nx = std::floor((b.x() - a.x()) / ElementLength) + 2; //Number of elements (intervals)
	double ny = std::floor((b.y() - a.y()) / ElementLength) + 2;
	mesh = Mesh(RectangleMesh(comm, a, b, nx, ny, "right")); //Rectangular mesh
	return 0;
}

int RectMeshGenerator(MPI_Comm comm, dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b) {
	mesh = Mesh(RectangleMesh(comm, a, b, 100, 100, "right")); //Rectangular mesh
	return 0;
}

int BoxMeshGenerator(MPI_Comm comm, dolfin::Mesh &mesh, dolfin::Point a, dolfin::Point b, double ElementLength) {
	auto nx = static_cast<unsigned long int>(std::floor((b.x() - a.x()) / ElementLength) + 2); //Number of elements (intervals)
	auto ny = static_cast<unsigned long int>(std::floor((b.y() - a.y()) / ElementLength) + 2);
	auto nz = static_cast<unsigned long int>(std::floor((b.z() - a.z()) / ElementLength) + 2);
	mesh = Mesh(BoxMesh(comm, a, b, nx, ny, nz)); //Box mesh
	return 0;
}

int BoxMeshGenerator(MPI_Comm comm, dolfin::Mesh mesh, dolfin::Point a, dolfin::Point b) {
	mesh = Mesh(BoxMesh(comm, a, b, 100, 100, 100)); //Box mesh
	return 0;
}

int myMeshRefiner(std::shared_ptr<dolfin::Mesh> mesh, std::vector<std::shared_ptr<dolfin::SubDomain>> myDomains) {
	dolfin::MeshFunction<bool> SubdomainMesh(mesh, 2);
	SubdomainMesh.set_all(false);
	for(std::size_t i=0; i<myDomains.size(); i=i+1)
		myDomains[i]->mark(SubdomainMesh, true);
	dolfin::Mesh refined_mesh;
	dolfin::refine(refined_mesh, *mesh, SubdomainMesh);
	*mesh = refined_mesh;
	return 0;
}

int myMeshRefiner(std::shared_ptr<dolfin::Mesh> mesh, std::shared_ptr<dolfin::SubDomain> myDomain) {
	dolfin::MeshFunction<bool> SubdomainMesh(mesh, 2);
	SubdomainMesh.set_all(false);
	myDomain->mark(SubdomainMesh, true);
	dolfin::Mesh refined_mesh;
	dolfin::refine(refined_mesh, *mesh, SubdomainMesh);
	*mesh = refined_mesh;
	return 0;
}

int myMeshRefiner(MPI_Comm comm, std::shared_ptr<dolfin::Mesh> mesh, std::vector<std::shared_ptr<dolfin::SubDomain>> myDomains) {
	dolfin::MeshFunction<bool> SubdomainMesh(mesh, 2);
	SubdomainMesh.set_all(false);
	for(std::size_t i=0; i<myDomains.size(); i=i+1)
		myDomains[i]->mark(SubdomainMesh, true);
	dolfin::Mesh refined_mesh(comm);
	dolfin::refine(refined_mesh, *mesh, SubdomainMesh);
	*mesh = refined_mesh;
	return 0;
}

int myMeshRefiner(MPI_Comm comm, std::shared_ptr<dolfin::Mesh> mesh, std::shared_ptr<dolfin::SubDomain> myDomain) {
	dolfin::MeshFunction<bool> SubdomainMesh(mesh, 2);
	SubdomainMesh.set_all(false);
	myDomain->mark(SubdomainMesh, true);
	dolfin::Mesh refined_mesh(comm);
	dolfin::refine(refined_mesh, *mesh, SubdomainMesh);
	*mesh = refined_mesh;
	return 0;
}

//b--c
//|  |
//a--d
//a-b:=width, a-d:=length
int RectPointsGenerator(double width, double length, std::vector<dolfin::Point> &ps) {
	ps.clear();
	ps.shrink_to_fit();

	//a
	ps.push_back(Point(1,1,0));
	//b
	ps.push_back(Point(1,1+width,0));
	//c
	ps.push_back(Point(1+length,1+width,0));
	//d
	ps.push_back(Point(1+length,1,0));

	return 0;
}

//		f--g
//roof side     |  |
//		e--h
//b--c
//|  |  floor side
//a--d
//a-b:=width, a-d:=length, a-e:=height
int BoxPointsGenerator(double width, double length, double height, std::vector<dolfin::Point> &ps) {
	ps.clear();
	ps.shrink_to_fit();

	//a
	ps.push_back(Point(1,1,1));
	//b
	ps.push_back(Point(1,1+width,1));
	//c
	ps.push_back(Point(1+length,1+width,1));
	//d
	ps.push_back(Point(1+length,1,1));

	//e
	ps.push_back(Point(1,1,1+height));
	//f
	ps.push_back(Point(1,1+width,1+height));
	//g
	ps.push_back(Point(1+length,1+width,1+height));
	//h
	ps.push_back(Point(1+length,1,1+height));

	return 0;
}

//to convert a mesh from xdmf to xml
int Mesh_XDMF2XML(std::string XDMFFileName="mesh.xdmf", std::string XMLFileName="mesh.xml") {

	dolfin::Mesh mesh;
	dolfin::XDMFFile XDMFStream(XDMFFileName);
	XDMFStream.read(mesh);

	dolfin::File XMLStream(XMLFileName);
	XMLStream<<(mesh);

	return 0;
}

