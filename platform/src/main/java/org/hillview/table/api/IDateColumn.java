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

package org.hillview.table.api;

import net.openhft.hashing.LongHashFunction;
import org.hillview.utils.Converters;

import javax.annotation.Nullable;
import java.time.Instant;

public interface IDateColumn extends IColumn {
    @Override
    default double asDouble(final int rowIndex, @Nullable final IStringConverter unused) {
        if (isMissing(rowIndex))
            throw new MissingException(this, rowIndex);
        final Instant tmp = this.getDate(rowIndex);
        return Converters.toDouble(Converters.checkNull(tmp));
    }

    @Nullable
    @Override
    default String asString(final int rowIndex) {
        if (this.isMissing(rowIndex))
            return null;
        return Converters.checkNull(this.getDate(rowIndex)).toString();
    }

    @Override
    default IndexComparator getComparator() {
        return new IndexComparator() {
            @Override
            public int compare(final Integer i, final Integer j) {
                final boolean iMissing = IDateColumn.this.isMissing(i);
                final boolean jMissing = IDateColumn.this.isMissing(j);
                if (iMissing && jMissing) {
                    return 0;
                } else if (iMissing) {
                    return 1;
                } else if (jMissing) {
                    return -1;
                } else {
                    return Converters.checkNull(IDateColumn.this.getDate(i))
                                     .compareTo(Converters.checkNull(IDateColumn.this.getDate(j)));
                }
            }
        };
    }

    @Override
    default long hashCode64(int rowIndex, LongHashFunction hash) {
        if (isMissing(rowIndex))
            return MISSING_HASH_VALUE;
        return hash.hashLong(Double.doubleToRawLongBits(this.asDouble(rowIndex, null)));
    }

    @Override
    default IColumn convertKind(ContentsKind kind, String newColName, IMembershipSet set) {
        IMutableColumn newColumn = this.allocateConvertedColumn(kind, set, newColName);
        switch(kind) {
            case Category:
            case Json:
            case String:
                //noinspection ConstantConditions
                this.convert(newColumn, set, row -> this.getDate(row).toString());
                break;
            case Date:
                this.convert(newColumn, set, this::getDate);
                break;
            case Integer:
            case Double:
            case Duration:
                throw new UnsupportedOperationException("Conversion from " + this.getKind()
                        + " to " + kind + " is not supported.");
            default:
                throw new RuntimeException("Unexpected column kind " + this.getKind());
        }
        return newColumn;
    }
 }
