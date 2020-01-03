#include "ModelManager.h"


int ModelManager::UpdateObjectInstances(KeyFrame* kf,
                                         std::vector<std::shared_ptr<SegData>> &SegDatas) 
{
    if (SegDatas.size() == 0)
        return -1;
    // no object in the manager
    if (mCurrentObjectIndex == 0)
    {
        for(auto segData:SegDatas)
        {
            BuildNewModel(kf->mnFrameId,segData);
        }
        return 0;
    }
    // 对于所有instance
    for(auto segData:SegDatas)
    {
        // 没有这个类别，直接加一个新的. 额外再判断一次，因为match 上不需要新建obj
        if(msObjectClasses.find(segData->classId) == msObjectClasses.end())
        {
            BuildNewModel(kf->mnFrameId,segData);
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
                float iou = CalculateIOUbyRoI(mTrackingInstances[ins_id]->GetLastRect(),segData->mImROI);
                if (iou>max_iou)
                {
                    max_iou = iou;
                    max_index = ins_id;
                }


            }
            if (max_iou>mMatchTHr)
            {
                mTrackingInstances[max_index]->UpdateObjectInfo(segData,kf->mnFrameId);
                //mTrackingInstances[max_index]->UpdateObjectInfo();
                //segData->instance_id = max_index;
            } 
            // 新建obj，有相同类别
            else
            {
                BuildNewModel(kf->mnFrameId,segData);
            }
        }

    }
    for(auto ins : mTrackingInstances)
    {
        if(ins.second->GetLastFrameID() != kf->mnFrameId)
        {
            mIndexLostnumMap[ins.second->GetObjIndex()] ++;
        } else
            mIndexLostnumMap[ins.second->GetObjIndex()] ==0;
    }

    for (auto item : mIndexLostnumMap)
    {
        if(item.second > mLostNumThr)
        {
            mDroppedInstances[item.first] = mTrackingInstances[item.first];
            mTrackingInstances.erase(item.first);
        }
    }
}

float ModelManager::CalculateIOUbyMask(const cv::Rect& a, const cv::Rect& b)
{

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

void ModelManager::BuildNewModel(const long unsigned int kf_index, const std::shared_ptr<SegData>& segData)
{
    std::cout<<"add now classes "<<segData->classId<<std::endl;
    std::shared_ptr<Model> newObjectInstance = std::make_shared<Model>(
                        mCurrentObjectIndex,kf_index,segData);
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
    std::unordered_map<int,std::shared_ptr<Model>>::iterator it;
    for(it=mTrackingInstances.begin();it!=mTrackingInstances.end();it++)
    {
        auto model=it->second;
        model->UpdatePointCloud(kf, color, depth, mask);
        *globalModel+=*(model->model);
    }
}