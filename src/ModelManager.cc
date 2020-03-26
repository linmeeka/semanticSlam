#include "ModelManager.h"


int ModelManager::UpdateObjectInstances(const KeyFrame* kf,
                                         const std::vector<std::shared_ptr<SegData>> &SegDatas,const cv::Mat &Mask) 
{
    if (SegDatas.size() == 0)
        return -1;
    // no object in the manager
    if (mCurrentObjectIndex == 0)
    {
        for(auto segData:SegDatas)
        {
            BuildNewModel(kf->mnFrameId,segData,Mask);
        }
        return 0;
    }
    
    // 对于所有instance
    std::unordered_map<int,std::shared_ptr<Model>>::iterator it;
    for(it=mTrackingInstances.begin();it!=mTrackingInstances.end();it++)
    {
        it->second->matched=false;
    }
    for(auto segData:SegDatas)
    {
        // 没有这个类别，直接加一个新的. 额外再判断一次，因为match 上不需要新建obj
        if(msObjectClasses.find(segData->classId) == msObjectClasses.end())
        {
            BuildNewModel(kf->mnFrameId,segData,Mask);
        } else
        // 有相同类别，计算IOU匹配
        {
            int max_index = 0;
            float max_iou = 0;
            for(int ins_id : mClassInstanceIdMap[segData->classId])
            {
                // 没在tracking，就跳过
                if (mTrackingInstances.find(ins_id) == mTrackingInstances.end())
                    continue;
                if(mTrackingInstances[ins_id]->matched==true)
                    continue;
                //float iou = CalculateIOUbyRoI(mTrackingInstances[ins_id]->GetLastRect(),segData->mImROI);
                float iou=CalculateIOUbyMask(mTrackingInstances[ins_id]->GetLastRect(),segData->mImROI,mTrackingInstances[ins_id]->GetLastMask(),Mask,segData->classId);
               
                if (iou>max_iou)
                {
                    max_iou = iou;
                    max_index = ins_id;
                }


            }
            float mMatchTHr;
            //if(mTrackingInstances[max_index]->isMoving)
            if(segData->weight!=0)
                mMatchTHr=mMatchTHr_moving;
            else
                mMatchTHr=mMatchTHr_static;
            if (max_iou>mMatchTHr)
            {
                mTrackingInstances[max_index]->UpdateObjectInfo(segData,kf->mnFrameId,Mask);
                //std::cout<<"model "<<max_index<<" matched with iou "<<max_iou<<" class id: "<<segData->classId<<std::endl;
                //mTrackingInstances[max_index]->UpdateObjectInfo();
                //segData->instance_id = max_index;
            } 
            // 新建obj，有相同类别
            else
            {
                BuildNewModel(kf->mnFrameId,segData,Mask);
            }
        }

    }
    for(auto ins : mTrackingInstances)
    {
        if(ins.second->GetLastFrameID() != kf->mnFrameId)
        {
            mIndexLostnumMap[ins.second->GetObjIndex()] ++;
        } else
            mIndexLostnumMap[ins.second->GetObjIndex()] =0;
    }

    for (auto item : mIndexLostnumMap)
    {
        if(mTrackingInstances.find(item.first)==mTrackingInstances.end())
            continue;            
        float mLostNumThr=1;
        //std::cout<<"is moving: "<<mTrackingInstances[item.first]->isMoving<<" class id :"<<mTrackingInstances[item.first]->GetClassId()<<std::endl;
        if(mTrackingInstances[item.first]->isMoving)
            mLostNumThr=mLostNumThr_moving;
        else
            mLostNumThr=mLostNumThr_static;
        if(item.second > mLostNumThr)
        {
            //if(mTrackingInstances.find(item.first)!=mTrackingInstances.end()) 
            //    std::cout<<"============model "<<item.first<<" lost, class id: "<<mTrackingInstances[item.first]->GetClassId()<<"=========="<<std::endl;
            //else
            //    std::cout<<"================= model lost but can not find it qaq =================="<<std::endl;
            mDroppedInstances[item.first] = mTrackingInstances[item.first];
            mTrackingInstances.erase(item.first);
            //mIndexLostnumMap.erase(item.first);
        }
    }
}

float ModelManager::CalculateIOUbyMask(const cv::Rect& roi1, const cv::Rect& roi2, const cv::Mat &mask1, const cv::Mat &mask2, const int &classId)
{
    cv::Mat img1=cv::Mat::zeros(mask1.size(), CV_8UC1); 
    cv::Mat r1=mask1(roi1).clone();
    //cv::imwrite("r10.jpg",r1);
    cv::threshold(r1,r1,classId,255,CV_THRESH_TOZERO_INV);
    //cv::imwrite("r11.jpg",r1);
    cv::threshold(r1,r1,classId-1,255,CV_THRESH_TOZERO);
    //cv::imwrite("r12.jpg",r1);
    cv::Mat imgroi1=img1(roi1);
    r1.copyTo(imgroi1,r1);
    //cv::imwrite("img1.jpg",img1);
    int t1=cv::countNonZero(img1);

    cv::Mat img2=cv::Mat::zeros(mask2.size(), CV_8UC1); 
    cv::Mat r2=mask1(roi2).clone();
    cv::threshold(r2,r2,classId,255,CV_THRESH_TOZERO_INV);
    cv::threshold(r2,r2,classId-1,255,CV_THRESH_TOZERO);
    cv::Mat imgroi2=img2(roi2);
    r2.copyTo(imgroi2,r2);
    //cv::imwrite("img2.jpg",img2);
    int t2=cv::countNonZero(img2);

    cv::bitwise_and(img1,img2,img1);

    //cv::imwrite("imgand.jpg",img1);
    int s1=cv::countNonZero(img1);
    int s2=t1+t2-s1;
    float res=(float)(s1)/(float)(s2);
    //std::cout<<"class id: "<<classId<<" iou "<<res<<" s1: "<<s1<<" s2: "<<s2<<endl;
    return res;
}

float ModelManager::CalculateIOUbyRoI(const cv::Rect& a, const cv::Rect& b)
{
    int x_min = a.x>b.x?a.x:b.x;
    int x_max = a.x+a.width>b.x+b.width?b.x+b.width:a.x+a.width;
    int y_min = a.y>b.y?a.y:b.y;
    int y_max = a.y+a.height>b.y+b.height?b.y+b.height:a.y+a.height;
    float i_area = 0;
    if (x_min < x_max && y_min < y_max)
        i_area = (x_max-x_min)*(y_max-y_min);
    if(a.area()+b.area()-i_area == 0)
        return 0;
    return i_area/(a.area()+b.area()-i_area);
}

void ModelManager::BuildNewModel(const long unsigned int kf_index, const std::shared_ptr<SegData>& segData,const cv::Mat &Mask)
{
    //std::cout<<"add new model "<<mCurrentObjectIndex<<" class id : "<<segData->classId<<std::endl;
    std::shared_ptr<Model> newObjectInstance = std::make_shared<Model>(
                        mCurrentObjectIndex,kf_index,segData,Mask);
    mTrackingInstances[mCurrentObjectIndex] = newObjectInstance;
    mIndexLostnumMap[mCurrentObjectIndex] = 0;
    //segData->instance_id = mCurrentObjectIndex;
    mCurrentObjectIndex ++;

    if(msObjectClasses.find(newObjectInstance->GetClassId())==msObjectClasses.end()) 
    {
        msObjectClasses.insert(newObjectInstance->GetClassId()); // class id -> instance id 
        std::vector<int> new_obj_set;
        new_obj_set.push_back(newObjectInstance->GetObjIndex());
        mClassInstanceIdMap[newObjectInstance->GetClassId()] = new_obj_set; // map <class id, set<instance id> >
    }
    else
        mClassInstanceIdMap[newObjectInstance->GetClassId()].push_back(newObjectInstance->GetObjIndex());
}

void ModelManager::UpdateObjectPointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, PointCloud::Ptr &globalModel)
{
    //cout<<" model num : "<<mTrackingInstances.size()<<endl;
    cv::Mat img=mask.clone();
    std::unordered_map<int,std::shared_ptr<Model>>::iterator it;
    for(it=mTrackingInstances.begin();it!=mTrackingInstances.end();it++)
    {
        auto model=it->second;
        //model->UpdatePointCloud(kf, color, depth, mask);
        model->UpdatePointCloud(kf, color, depth, img);
        *globalModel+=*(model->model);
    }
    //cv::imwrite("img.jpg",img);
}