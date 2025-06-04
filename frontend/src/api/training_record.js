import axios from "@/axios"

/**
 * 提交训练记录
 * @param {Object} data 训练数据
 * @param {string} data.trainingDate - 训练日期时间，格式：yyyy-MM-dd'T'HH:mm:ss
 * @param {number} data.duration - 训练时长（分钟）
 * @param {number} data.distance - 训练距离（公里）
 * @param {number|null} data.calories - 消耗卡路里（可选）
 * @param {number} data.status - 训练状态：0-进行中，1-已完成
 * @param {string} data.type - 训练类型
 * @returns {Promise} 返回训练记录ID
 */
export function submitTrainingRecord(data) {
    // 格式化提交数据
    const submitData = {
        trainingDate: formatDateToISOString(data.trainingDate),
        duration: parseInt(data.duration, 10),
        distance: parseFloat(data.distance),
        calories: data.calories ? parseFloat(data.calories) : null,
        status: data.status === undefined ? 1 : parseInt(data.status, 10),
        type: data.type
    }

    return axios.post("/club/training-data/submit", submitData)
        .then(response => {
            console.log('Server response:', response)

            if (!response || !response.data) {
                throw new Error('服务器响应为空')
            }

            // 如果响应是对象且包含success字段，返回整个响应
            if (typeof response.data === 'object' && 'success' in response.data) {
                return response.data
            }
            
            // 如果响应是字符串（ID），包装成成功响应
            return {
                success: true,
                data: response.data,
                message: '添加成功'
            }
        })
        .catch(error => {
            console.error('Training record submission error:', error)
            if (error.response) {
                const { data, status } = error.response
                throw new Error(data?.message || `服务器错误(${status})`)
            }
            throw error
        });
}

/**
 * 查询训练记录列表
 * @param {Object} params 查询参数
 * @param {number} params.page - 页码
 * @param {number} params.size - 每页数量
 * @param {string} [params.type] - 训练类型（可选）
 * @param {string} [params.startDate] - 开始日期，格式：YYYY-MM-DD（可选）
 * @param {string} [params.endDate] - 结束日期，格式：YYYY-MM-DD（可选）
 * @returns {Promise} 返回训练记录列表和分页信息
 */
export function queryTrainingRecords(params) {
    // 构建查询参数，移除所有 undefined 和 null 值
    const queryParams = Object.entries({
        page: params.page || 1,
        size: params.size || 10,
        type: params.type,
        startDate: params.startDate,
        endDate: params.endDate
    }).reduce((acc, [key, value]) => {
        if (value !== undefined && value !== null) {
            acc[key] = value
        }
        return acc
    }, {})

    console.log('Query params:', queryParams)

    return axios.post("/club/training-data/query", queryParams)
        .then(response => {
            console.log('Server response:', response)
            if (!response) {
                throw new Error('服务器响应为空')
            }
            // 直接返回响应数据，因为axios拦截器已经处理了data字段
            return response
        })
        .catch(error => {
            console.error("Error querying training records:", error)
            throw error.response?.data?.message || error.message || '查询失败'
        })
}

/**
 * 更新训练记录
 * @param {Object} data 更新数据
 * @param {string|number} data.id - 训练记录ID
 * @param {string} [data.trainingDate] - 训练日期时间
 * @param {number} [data.duration] - 训练时长（分钟）
 * @param {number} [data.distance] - 训练距离（公里）
 * @param {number} [data.calories] - 消耗卡路里
 * @param {string} [data.type] - 训练类型
 * @returns {Promise}
 */
export function updateTrainingRecord(data) {
    // 格式化更新数据
    const updateData = {
        id: data.id,
        ...(data.trainingDate && { trainingDate: formatDateToISOString(data.trainingDate) }),
        ...(data.duration && { duration: parseInt(data.duration, 10) }),
        ...(data.distance && { distance: parseFloat(data.distance) }),
        ...(data.calories && { calories: parseFloat(data.calories) }),
        ...(data.type && { type: data.type })
    }

    // 数据验证
    if (updateData.duration && updateData.duration <= 0) {
        return Promise.reject(new Error('训练时长必须是正数'))
    }
    if (updateData.distance && updateData.distance <= 0) {
        return Promise.reject(new Error('训练距离必须是正数'))
    }
    if (updateData.calories && updateData.calories < 0) {
        return Promise.reject(new Error('卡路里不能为负数'))
    }

    return axios.post("/club/training-data/update", updateData)
        .then(response => {
            if (!response.data) {
                throw new Error('服务器响应为空')
            }
            if (!response.data.success) {
                throw new Error(response.data.message || '更新失败')
            }
            return response.data
        })
        .catch(error => {
            console.error("Error updating training record:", error);
            throw error.response?.data?.message || error.message || '更新失败'
        });
}

/**
 * 删除训练记录
 * @param {string|number} id 训练记录ID
 * @returns {Promise} 返回删除操作的结果
 */
export function deleteTrainingRecord(id) {
    return axios.post("/club/training-data/delete", { id })
        .then(response => {
            if (!response) {
                throw new Error('服务器响应为空')
            }
            if (!response.success) {
                throw new Error(response.message || '删除失败')
            }
            return response
        })
        .catch(error => {
            console.error("Error deleting training record:", error)
            throw error.response?.data?.message || error.message || '删除失败'
        })
}

/**
 * 获取训练统计数据
 * @param {Object} params 查询参数
 * @param {string} params.type - 训练类型
 * @param {string} [params.startDate] - 开始日期，格式：YYYY-MM-DD（可选）
 * @param {string} [params.endDate] - 结束日期，格式：YYYY-MM-DD（可选）
 * @returns {Promise} 返回统计数据
 */
export function getTrainingStatistics(params) {
    // 构建查询参数，移除所有 undefined 和 null 值
    const queryParams = Object.entries({
        type: params.type,
        startDate: params.startDate,
        endDate: params.endDate
    }).reduce((acc, [key, value]) => {
        if (value !== undefined && value !== null) {
            acc[key] = value
        }
        return acc
    }, {})

    console.log('Statistics query params:', queryParams)

    return axios.post("/club/training-data/statistics", queryParams)
        .then(response => {
            console.log('Statistics response:', response)
            if (!response) {
                throw new Error('服务器响应为空')
            }
            return response
        })
        .catch(error => {
            console.error("Error getting training statistics:", error)
            throw error.response?.data?.message || error.message || '获取统计数据失败'
        })
}

/**
 * 格式化日期为ISO字符串
 * @param {string|Date} date - 日期
 * @returns {string} 格式化后的日期字符串 yyyy-MM-dd'T'HH:mm:ss
 */
function formatDateToISOString(date) {
    if (!date) return null;
    
    let dateObj;
    if (typeof date === 'string') {
        // 处理已经是ISO格式的情况
        if (date.includes('T')) {
            return date;
        }
        // 处理其他格式的日期字符串
        dateObj = new Date(date.replace(' ', 'T'));
    } else if (date instanceof Date) {
        dateObj = date;
    } else {
        return null;
    }

    // 检查日期是否有效
    if (isNaN(dateObj.getTime())) {
        return null;
    }

    // 格式化为后端要求的格式
    const isoString = dateObj.toISOString().slice(0, 19);
    console.log('Formatted date:', isoString); // 调试日期格式
    return isoString;
}
