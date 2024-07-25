//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Project includes
// External includes
#include <Metal/Metal.hpp>
// System includes
#include <map>


namespace aix
{

class MTLBufferCache
{
public:
    // Constructor
    MTLBufferCache() = default;

    // Destructor
    ~MTLBufferCache()  { clear(); }

    inline size_t size() const  { return m_cacheSize; }

    void clear()
    {
        for (auto& [size, cacheBufHolder] : m_cacheMap)
        {
            if (cacheBufHolder->buffer)
            {
                cacheBufHolder->buffer->release();
            }
            delete cacheBufHolder;
        }
        m_cacheMap.clear();
        m_bhHead = m_bhTail = nullptr;
        m_cacheSize = 0;
    }

    MTL::Buffer* reuse(size_t size)
    {
        MTL::Buffer* buffer{nullptr};

        // Find the closest buffer in the cached buffers with a similar size.
        auto it = m_cacheMap.lower_bound(size);

        // Make sure we use most of the available memory.
        while (!buffer && it != m_cacheMap.end() && it->first < std::min(2 * size, size + 2 * ALLOCATION_BYTE_ALIGNMENT_SIZE))
        {
            // Collect from the cache.
            buffer = it->second->buffer;

            // Remove from the cache.
            removeFromList(it->second);
            delete it->second;
            it = m_cacheMap.erase(it);
        }

        if (buffer)
        {
            m_cacheSize -= buffer->length();
        }

        return buffer;
    }

    void recycle(MTL::Buffer* buffer)
    {
        if (!buffer) return;
        // Add to the cache.
        auto bufHolder = new BufferHolder(buffer);
        addAtHead(bufHolder);
        m_cacheSize += buffer->length();
        m_cacheMap.insert({buffer->length(), bufHolder});
    }

    void reduceSize(size_t bytesToFree)
    {
        if (bytesToFree < static_cast<size_t>(m_cacheSize * 0.9f))
        {
            size_t totalBytesFreed = 0;
            while (m_bhTail && (totalBytesFreed < bytesToFree))
            {
                if (m_bhTail->buffer)
                {
                    totalBytesFreed += m_bhTail->buffer->length();
                    m_bhTail->buffer->release();
                    m_bhTail->buffer = nullptr;
                }
                removeFromList(m_bhTail);
            }
            m_cacheSize -= totalBytesFreed;
        }
        else
        {
            clear();
        }
    }

private:
    struct BufferHolder
    {
        BufferHolder(MTL::Buffer* buffer) : buffer(buffer) { }
        MTL::Buffer*  buffer;
        BufferHolder* prev{nullptr};
        BufferHolder* next{nullptr};
    };

    void addAtHead(BufferHolder* bufHolder)
    {
        if (!bufHolder) return;

        if (!m_bhHead)
        {
            m_bhHead = m_bhTail = bufHolder;
        }
        else
        {
            m_bhHead->prev = bufHolder;
            bufHolder->next = m_bhHead;
            m_bhHead = bufHolder;
        }
    }

    void removeFromList(BufferHolder* bufHolder)
    {
        if (!bufHolder)  return;

        // If in the middle.
        if (bufHolder->prev && bufHolder->next)
        {
            bufHolder->prev->next = bufHolder->next;
            bufHolder->next->prev = bufHolder->prev;
        }
        else if (bufHolder->prev && bufHolder == m_bhTail)
        {   // If tail.
            m_bhTail = bufHolder->prev;
            m_bhTail->next = nullptr;
        }
        else if (bufHolder == m_bhHead && bufHolder->next)
        {   // If head.
            m_bhHead = bufHolder->next;
            m_bhHead->prev = nullptr;
        }
        else if (bufHolder == m_bhHead && bufHolder == m_bhTail)
        {   // If only element.
            m_bhHead = m_bhTail = nullptr;
        }

        bufHolder->prev = bufHolder->next = nullptr;
    }

    std::multimap<size_t, BufferHolder*> m_cacheMap;
    BufferHolder* m_bhHead{nullptr};
    BufferHolder* m_bhTail{nullptr};
    size_t m_cacheSize{0};
};

}   // namespace aix
