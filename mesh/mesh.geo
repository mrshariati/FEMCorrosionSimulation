h = 0.0001;
//+
Point(1) = {1, 1, 0, h};
//+
Point(2) = {1, 1.01, 0, 10*h};
//+
Point(3) = {1.02, 1.01, 0, 10*h};
//+
Point(4) = {1.02, 1, 0, h};
//+
Point(5) = {1.006, 1, 0, 0.2*h};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
