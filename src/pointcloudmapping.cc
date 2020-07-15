/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "pointcloudmapping.h"
#include "segmentation.h"
#include <Eigen/Geometry>
#include <KeyFrame.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <ctime>
#include <pcl/surface/gp3.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "Converter.h"
#include <boost/make_shared.hpp>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>





using namespace cv;
using namespace std;
using namespace pcl;
using namespace APC;
using namespace pcl::io;
using namespace pcl::console;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;


int pub_port = 6666;
#define NUM 5


int k = 500;
int min_size = 500;


PointCloudMapping::PointCloudMapping(double resolution_) {

    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();
    background = boost::make_shared<PointCloud>();
    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));
    int max_obj_num = 100;
    int max_lost_num = 5;
    //float match_thr = 0.95;
    modelManager = ModelManager(max_obj_num, max_lost_num);

}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, std::vector<std::shared_ptr<SegData>> &segDatas)
{


    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;

    unique_lock<mutex> lck(keyframeMutex);
    // keyframes.push_back( kf );
    // colorImgs.push_back( color.clone());
    // depthImgs.push_back( depth.clone());
    // maskImgs.push_back(mask.clone());
    keyframes.push( kf );
    colorImgs.push( color.clone());
    depthImgs.push( depth.clone());
    maskImgs.push(mask.clone());
    segDataQue.push(segDatas);
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    // 3*3的像素区域取一个点
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            int val=mask.ptr<uchar>(m)[n];
            if(val!=0)
                continue;
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
            tmp->points.push_back(p);
        }
    }
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}
    int sumKF=0;
    double sumTime=0;
void PointCloudMapping::viewer()
{
    return ;
    std::cout<<"enter viewer "<<std::endl;
    sleep(3);
    pcl::visualization::CloudViewer viewer("viewer");

    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }

        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }


        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }


        {
            for(int i=0;i<N;i++)
            //for (size_t i=lastKeyframeSize; i<N ; i++) 
            {
                                #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                #else
                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
                #endif
                // 为每个KF生成点云
                //PointCloud::Ptr surf_p = generatePointCloud(keyframes[i], img_tmp_color, depthImgs[i]);
                KeyFrame* kf=keyframes.front();
                keyframes.pop();
                cv::Mat colorImg=colorImgs.front();
                colorImgs.pop();
                cv::Mat depthImg=depthImgs.front();
                depthImgs.pop();
                cv::Mat maskImg=maskImgs.front();
                maskImgs.pop();
                std::vector<std::shared_ptr<SegData>> segDatas=segDataQue.front();
                segDataQue.pop();
                int kfid=kf->mnId;
                cout<<"=====debug=====: get kf from queue: "<<kfid<<endl;
                // 更新背景
                PointCloud::Ptr surf_p = generatePointCloud(kf, colorImg, depthImg,maskImg);
                *background += *surf_p;
                *globalMap=*background;
                //cout<<"=====debug=====: update background "<<endl;
                 // 在这里调用model maneger 更新model
                modelManager.UpdateObjectInstances(kf,segDatas,maskImg);
                //cout<<"=====debug=====: update instance"<<endl;
                cout<<" segData size: "<<segDatas.size()<<endl;
                modelManager.UpdateObjectPointCloud(kf, colorImg, depthImg,maskImg,globalMap);
                //cout<<"=====debug=====: update moving object"<<endl;
                //PointCloud::Ptr p = RegionGrowingSeg(surf_p); 
                #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
         #else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
        #endif
        sumKF++;
        double ttrack1= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        sumTime+=ttrack1;
        voxel.setInputCloud( globalMap );
        viewer.showCloud( globalMap );
        cout << "show global map, size=" << globalMap->points.size() << endl;
        cout << "ttrack1 " << ttrack1 << endl;
        lastKeyframeSize = N;               

            }
        }
        
    }
    std::cout<<"map per kf: "<<sumTime/sumKF<<std::endl;
    std::cout<<"total num of kf: "<<sumKF<<std::endl;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PCDWriter pcdwriter;
    // pcdwriter.write<pcl::PointXYZRGBA>("global_color.pcd", *globalMap);//write global point cloud map and save to a pcd file
    // cpf_seg(globalMap);
   
}
