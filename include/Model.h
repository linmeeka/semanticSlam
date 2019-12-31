/**
* This file is added by kylin
*/

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include "SegData.h"
#include "Frame.h"
#include "KeyFrame.h"
#include <map>
#include <set>
#include <unordered_map>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <condition_variable>
//using namespace std;
using namespace ORB_SLAM2;

class Model
{
public:
    // explicit Model(const int index);
    explicit Model(const int index, const long unsigned int kf_index, const std::shared_ptr<SegData>& segData);
    // explicit Model(const int index,const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo, \
    // const long unsigned int kf_index,const std::vector<MapPoint*>& pvMapPoints);

    // void UpdateObjectInfo(const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo, \
    // const long unsigned int kf_index,const std::vector<MapPoint*>&pvMapPoints);

    void UpdateObjectInfo(const std::shared_ptr<SegData>& segData, const long unsigned int kf_index);
    //void UpdateObjectInfo()

    //Model();
    //void UpdateModel();

    int GetClassId()
    {
        return mClassId;
    }
    
    int GetObjIndex()
    {
        return mIndex;
    }
    cv::Rect GetLastRect()
    {
        if(mvKeyframeIndexes.size() > 0)
        {
            return mmKeyFrameObjectInfo[mvKeyframeIndexes[mvKeyframeIndexes.size()-1]]->mImROI;
        }
    }
    PointCloud::*Ptr getPointCloudModel()
    {
        return *model;
    }
    cv::Rect GetRoI()
    {

    }
    long unsigned int GetLastFrameID()
    {
        if(mvKeyframeIndexes.size()>0)
            return mvKeyframeIndexes[mvKeyframeIndexes.size()-1];
    }
    void UpdatePointCloud(const KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask);

private:
    int mIndex;
    int mClassId;
    std::vector<long unsigned int> mvKeyframeIndexes; // KFs that observed this model
    std::set<long unsigned  int>mvMapPointIndexes;
    std::unordered_map<long unsigned int,std::shared_ptr<SegData>> mmKeyFrameObjectInfo; //KFIndex -> Objs
    //std::unordered_map<long unsigned int,std::shared_ptr<ImgObjectInfo>> mmKeyFrameObjectInfo;
    PointCloud::Ptr model;
    int isMoving;

    void GetIncrementModel(const KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask, PointCloud::Ptr &inc);
};



#endif