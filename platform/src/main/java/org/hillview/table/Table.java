/*
 * Copyright (c) 2017 VMware Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.hillview.table;

import org.hillview.table.api.*;
import org.hillview.utils.Utilities;

/**
 * This is a simple table held entirely in RAM.
 */
public class Table extends BaseTable {
    private final Schema schema;
    private final IMembershipSet members;

    /**
     * Create an empty table with the specified schema.
     * @param schema schema of the empty table
     */
    public Table(final Schema schema) {
        super(schema);
        this.schema = schema;
        this.members = new FullMembership(0);
    }

    /**
     * Create a table from raw ingredients.
     * @param columns  Columns in the table.
     * @param members  Membership set (rows in the table).
     * @param schema   Schema; must match the set of columns.
     */
    protected <C extends IColumn> Table(
            final Iterable<C> columns, final IMembershipSet members, Schema schema) {
        super(columns);
        this.members = members;
        this.schema = schema;
    }

    public <C extends IColumn> Table(final Iterable<C> columns, final IMembershipSet members) {
        super(columns);
        final Schema s = new Schema();
        for (final IColumn c : columns)
            s.append(c.getDescription());
        this.schema = s;
        this.members = members;
    }

    public <C extends IColumn> Table(final Iterable<C> columns) {
        this(columns, new FullMembership(columnSize(columns)));
    }

    public <C extends IColumn> Table(final C[] columns) {
        this(Utilities.arrayToIterable(columns));
    }

    public <C extends IColumn> Table(final C[] columns, IMembershipSet set) {
        this(Utilities.arrayToIterable(columns), set);
    }

    @Override
    public ITable project(Schema schema) {
        IColumn[] cols = this.getColumns(schema);
        return new Table(cols, this.members);
    }

    @Override
    public ITable replace(IColumn[] columns) {
        return new Table(columns, this.getMembershipSet());
    }

    @Override
    public ColumnAndConverter[] getLoadedColumns(ColumnAndConverterDescription[] columns) {
        ColumnAndConverter[] result = new ColumnAndConverter[columns.length];
        for (int i=0; i < columns.length; i++) {
            String name = columns[i].columnName;
            IColumn col = this.columns.get(name);
            result[i] = new ColumnAndConverter(col, columns[i].getConverter());
        }
        return result;
    }

    @Override
    public Schema getSchema() {
        return this.schema;
    }

    @Override
    public IRowIterator getRowIterator() {
        return this.members.getIterator();
    }

    /**
     * Describes the set of rows that are really present in the table.
     */
    @Override
    public IMembershipSet getMembershipSet() { return this.members; }

    @Override
    public int getNumOfRows() {
        return this.members.getSize();
    }

    @Override
    public ITable selectRowsFromFullTable(IMembershipSet set) {
        return new Table(this.getColumns(), set);
    }

    /**
     * Generates a table that contains all the columns, and only
     * the rows contained in IMembership Set members with consecutive numbering.
     */
    public SmallTable compress() {
        final ISubSchema subSchema = new FullSubSchema();
        return this.compress(subSchema, this.members);
    }
}
