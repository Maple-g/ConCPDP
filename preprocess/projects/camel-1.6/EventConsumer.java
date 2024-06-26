/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.component.event;

import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.impl.DefaultConsumer;

/**
 * An <a href="http://activemq.apache.org/camel/event.html">Event Consumer</a>
 * for working with Spring ApplicationEvents
 *
 * @version $Revision: 673477 $
 */
public class EventConsumer extends DefaultConsumer<Exchange> {
    private EventEndpoint endpoint;

    public EventConsumer(EventEndpoint endpoint, Processor processor) {
        super(endpoint, processor);
        this.endpoint = endpoint;
    }

    @Override
    protected void doStart() throws Exception {
        super.doStart();
        endpoint.consumerStarted(this);
    }

    @Override
    protected void doStop() throws Exception {
        endpoint.consumerStopped(this);
        super.doStop();
    }
}
