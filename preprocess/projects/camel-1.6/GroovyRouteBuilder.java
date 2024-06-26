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
package org.apache.camel.language.groovy;

import groovy.lang.GroovyShell;
import org.apache.camel.CamelContext;
import org.apache.camel.builder.RouteBuilder;

/**
 * @version $Revision: 640731 $
 */
public abstract class GroovyRouteBuilder extends RouteBuilder {
    public GroovyRouteBuilder() {
        init();
    }

    public GroovyRouteBuilder(CamelContext context) {
        super(context);
        init();
    }

    private void init() {
        ClassLoader loader = getClass().getClassLoader();
        GroovyShell shell = new GroovyShell(loader);
        shell.evaluate(loader.getResourceAsStream("org/apache/camel/language/groovy/ConfigureCamel.groovy"));

        // TODO compile Groovy as part of build!
        //new ConfigureCamel().run();
    }
}
