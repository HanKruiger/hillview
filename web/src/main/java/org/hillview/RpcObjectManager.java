/*
 * Copyright (c) 2017 VMware Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.hillview;

import org.hillview.utils.HillviewLogger;
import rx.Observer;
import rx.Subscription;

import javax.annotation.Nullable;
import javax.websocket.Session;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * The RpcObjectManager manages a pool of objects that are the targets of RPC calls
 * from the clients.  These are RpcTarget objects, and each one has a unique
 * identifier.  This class manages these identifiers and keeps track of the mapping
 * between identifiers and objects.
 *
 * The class also keeps track of open sessions and matches sessions to RpcTargets.
 *
 * This is a singleton pattern.
 */
public final class RpcObjectManager {
    /**
     * Well-known id of the initial object.
     */
    static final String initialObjectId = "0";

    // We have exactly one instance of this object, because the web server
    // is multi-threaded and it instantiates various classes on demand to service requests.
    // These need to be able to find the ObjectManager - they do it through
    // the unique global instance.
    public static final RpcObjectManager instance;

    // Map object id to object.
    private final HashMap<String, RpcTarget> objects;

    // Map the session to the targetId object that is replying to the request, if any.
    private final HashMap<Session, RpcTarget> sessionRequest =
            new HashMap<Session, RpcTarget>(10);
    // Mapping sessions to RxJava subscriptions - needed to do cancellations.
    private final HashMap<Session, Subscription> sessionSubscription =
            new HashMap<Session, Subscription>(10);

    // TODO: persist object history into persistent storage.
    // For each object id the computation that has produced it.
    private final HashMap<String, HillviewComputation> generator =
            new HashMap<String, HillviewComputation>();

    synchronized void addSession(Session session, @Nullable RpcTarget target) {
        this.sessionRequest.put(session, target);
    }

    synchronized void removeSession(Session session) {
        this.sessionRequest.remove(session);
    }

    @Nullable synchronized RpcTarget getTarget(Session session) {
        return this.sessionRequest.get(session);
    }

    @Nullable synchronized Subscription getSubscription(Session session) {
        return this.sessionSubscription.get(session);
    }

    synchronized void addSubscription(Session session, Subscription subscription) {
        if (subscription.isUnsubscribed())
            // The computation may have already finished by the time we get here!
            return;
        HillviewLogger.instance.info("Saving subscription", "{0}", this.toString());
        if (this.sessionSubscription.get(session) != null)
            throw new RuntimeException("Subscription already active on this context");
        this.sessionSubscription.put(session, subscription);
    }

    synchronized void removeSubscription(Session session) {
        HillviewLogger.instance.info("Removing subscription", "{0}", this.toString());
        this.sessionSubscription.remove(session);
    }

    static {
        instance = new RpcObjectManager();
        new InitialObjectTarget();  // indirectly registers this object with the RpcObjectManager
    }

    // Private constructor
    private RpcObjectManager() {
        this.objects = new HashMap<String, RpcTarget>();
    }

    synchronized void addObject(RpcTarget object) {
        if (this.objects.containsKey(object.objectId))
            throw new RuntimeException("Object with id " + object.objectId + " already in map");
        HillviewLogger.instance.info("Object generated", "{0} from {1}", object.objectId, object.computation);
        this.generator.put(object.objectId, object.computation);
        HillviewLogger.instance.info("Inserting targetId", "{0}", object.toString());
        this.objects.put(object.objectId, object);
    }

    synchronized @Nullable RpcTarget getObject(String id) {
        HillviewLogger.instance.info("Getting object", "{0}", id);
        return this.objects.get(id);
    }

    /**
     * Attempt to retrieve the object with the specified id.
     * @param id           Object id to retrieve.
     * @param toNotify     An observer notified when the object is retrieved.
     * @param rebuild      If true attempt to rebuild the object if not found.
     */
    void retrieveTarget(String id, boolean rebuild, Observer<RpcTarget> toNotify) {
        RpcTarget target = this.getObject(id);
        if (target != null) {
            toNotify.onNext(target);
            toNotify.onCompleted();
            return;
        }
        if (rebuild) {
            this.rebuild(id, toNotify);
        } else {
            toNotify.onError(new RuntimeException("Cannot find object " + id));
        }
    }

    /**
     * We have lost the object with the specified id.  Try to reconstruct it
     * from the history.
     * @param id  Id of object to reconstruct.
     * @param toNotify An observert that is notified when the object is available.
     */
    private void rebuild(String id, Observer<RpcTarget> toNotify) {
        HillviewLogger.instance.info("Attempt to reconstruct", "{0}", id);
        HillviewComputation computation = this.generator.get(id);
        if (computation != null) {
            // The following may trigger a recursive reconstruction.
            HillviewLogger.instance.info("Replaying", "{0}", computation);
            computation.replay(toNotify);
        } else {
            Exception ex = new RuntimeException("Cannot reconstruct " + id);
            HillviewLogger.instance.error("Could not locate computation", ex);
            toNotify.onError(ex);
        }
    }

    @SuppressWarnings("unused")
    synchronized private void deleteObject(String id) {
        if (!this.objects.containsKey(id))
            throw new RuntimeException("Object with id " + id + " does not exist");
        this.objects.remove(id);
    }

    /**
     * Removes all RemoteObjects from the cache, except the initial object.
     * @return  The number of objects removed.
     */
    int removeAllObjects() {
        List<String> toDelete = new ArrayList<String>();
        for (String k: this.objects.keySet()) {
            if (!k.equals(initialObjectId))
                toDelete.add(k);
        }

        for (String k: toDelete)
            this.deleteObject(k);
        return toDelete.size();
    }
}
