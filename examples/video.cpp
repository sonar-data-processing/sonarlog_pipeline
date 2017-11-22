#include<iostream>
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include <opencv2/core/core.hpp>
 #include<dirent.h>
 #include<string.h>

 using namespace std;
 using namespace cv;

 int main()
 {
     string dirName = "/home/romulo/workspace/sonar_toolkit/sonarlog_pipeline/frames/";
     DIR *dir;
     dir = opendir(dirName.c_str());
     string imgName;
     struct dirent *ent;
     if (dir != NULL) {
     while ((ent = readdir (dir)) != NULL) {
           imgName= ent->d_name;
           //I found some . and .. files here so I reject them.
           if(imgName.compare(".")!= 0 && imgName.compare("..")!= 0)
           {
             string aux;
             aux.append(dirName);
             aux.append(imgName);
             cout << aux << endl;
             Mat image= imread(aux);
             imshow(aux,image);
             waitKey(0);
           }
      }
      closedir (dir);
  } else {
      cout<<"not present"<<endl;
     }
 }
