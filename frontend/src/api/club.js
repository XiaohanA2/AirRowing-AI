import axios from "@/axios";

// 俱乐部基础操作
export function createClubService(params) {
    console.log('Creating club with params:', params);
    return axios.post("/club/club/create", params)
        .then(response => {
            console.log('Create club response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error creating club:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function updateClubService(params) {
    console.log('Updating club with params:', params);
    return axios.post("/club/club/update", params)
        .then(response => {
            console.log('Update club response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error updating club:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function deleteClubService(params) {
    console.log('Deleting club with params:', params);
    return axios.post("/club/club/delete", params)
        .then(response => {
            console.log('Delete club response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error deleting club:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function getClubDetailsService(params) {
    console.log('Requesting club details with params:', params);
    return axios.post("/club/club/details", params)
        .then(response => {
            console.log('Club details response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching club details:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 获取俱乐部列表
export function getClubListService(params) {
    console.log('Requesting club list with params:', params);
    return axios.post("/club/club/clublist", params)
        .then(response => {
            console.log('Club list response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching club list:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 俱乐部成员管理
export function addSecondHostService(params) {
    return axios.post("/club/addHosts", params);
}

export function removeHostService(params) {
    return axios.post("/club/firehost", params);
}

export function getHostListService(params) {
    return axios.post("/club/hostlist", params);
}

export function getClubMembersService(params) {
    console.log('Fetching club members with params:', params);
    return axios.post("/club/club/members", params)
        .then(response => {
            console.log('Club members response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching club members:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function removeMemberService(params) {
    console.log('Removing club member with params:', params);
    return axios.post("/club/club/clubRemoveMember", params)
        .then(response => {
            console.log('Remove member response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error removing member:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 加入申请相关
export function joinClubService(params) {
    console.log('Submitting join request with params:', params);
    return axios.post("/club/club/joinRequest", params)
        .then(response => {
            console.log('Join request response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error submitting join request:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function reviewJoinRequestService(params) {
    console.log('Reviewing join request with params:', params);
    return axios.post("/club/club/reviewJoinRequest", params)
        .then(response => {
            console.log('Review join request response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error reviewing join request:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 退出俱乐部
export function leaveClubService(params) {
    console.log('Leaving club with params:', params);
    return axios.post("/club/club/leave", params)
        .then(response => {
            console.log('Leave club response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error leaving club:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 获取加入申请列表
export function getJoinRequestListService(params) {
    return axios.post("/club/joinRequestList", params)
        .then(response => response.data)
        .catch(error => {
            console.error("Error fetching join requests:", error);
            throw error;
        });
}

export function getClubJoinRequestsService(params) {
    console.log('Fetching club join requests with params:', params);
    return axios.post("/club/club/clubJoinRequests", params)
        .then(response => {
            console.log('Club join requests response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching club join requests:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 活动相关
export function publishActivityService(params) {
    console.log('Publishing activity with params:', params);
    return axios.post("/club/club/publishActivity", params)
        .then(response => {
            console.log('Publish activity response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error publishing activity:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function updateActivityService(params) {
    return axios.post("/club/updateActivity", params);
}

export function deleteActivityService(params) {
    return axios.post("/club/deleteActivity", params);
}

export function getActivityDetailsService(params) {
    console.log('Fetching activity details with params:', params);
    return axios.post("/club/club/activity/details", params)
        .then(response => {
            console.log('Activity details response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching activity details:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

export function getClubActivityListService(params) {
    console.log('Fetching activity list with params:', params);
    return axios.post("/club/club/activityList", params)
        .then(response => {
            console.log('Activity list response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching activity list:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}

// 活动报名相关
export function signupActivityService(params) {
    return axios.post("/club/activity/signup", params);
}

export function reviewSignupService(params) {
    return axios.post("/club/activity/reviewSignup", params);
}

export function cancelSignupService(params) {
    return axios.post("/club/activity/cancelSignup", params);
}

export function removeActivityMemberService(params) {
    return axios.post("/club/activity/removeMember", params);
}

export function getActivityMembersService(params) {
    return axios.post("/club/activity/signupMembers", params)
        .then(response => response.data)
        .catch(error => {
            console.error("Error fetching activity members:", error);
            throw error;
        });
}

export function getUserSignupsService(params) {
    return axios.post("/club/activity/userSignups", params)
        .then(response => response.data)
        .catch(error => {
            console.error("Error fetching user signups:", error);
            throw error;
        });
}

export function getHostSignupListService(params) {
    return axios.post("/club/activity/hostSignupList", params)
        .then(response => response.data)
        .catch(error => {
            console.error("Error fetching host signup list:", error);
            throw error;
        });
} 