export default class ScatterplotGrid {
    constructor(
        container,
        legendContainer,
        plotSize,
        pointSize,
        nPlots = 3) {
        this.container = container;
        this.legendContainer = legendContainer;
        // this._plotSize = plotSize;
        this._pointSize = pointSize;
        this.nPlots = nPlots;

        this.xScale;
        this.yScale;
        this.colorScale;

        this.width = plotSize;
        this.height = plotSize;
        this.margin = { top: 10, right: 10, bottom: 10, left: 10 }
        this.svgs = [];

        this.tooltip = d3.select("body")
            .append("div")
            .style("position", "absolute")
            .style("z-index", "10")
            .style("visibility", "hidden")
            .style("background", "#DDD")
            .style("padding", "0pt 4pt")
            .text("");

        for (let i = 0; i < nPlots; i++) {
            this._createScatterplotWithSlider(i);
        }
    }

    changePointSize(size) {
        this._pointSize = size;
        d3.select("#container").selectAll("circle")
            .attr("r", size)
    }

    changePlotSize(size) {
        this.width = size;
        this.height = size;

        // adjust scales
        this.xScale.range([this.margin.left, this.width - this.margin.right]);
        this.yScale.range([this.height - this.margin.bottom, this.margin.top]);
        // adjust svg size and data point positions
        d3.selectAll("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .selectAll("circle")
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y));;

        // reset grid lines
        this._removeGridLines();

        // reset zoom
        this._resetZoom();

    }

    changeCategory(category) {
        // resest color scale
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        // change data point color
        d3.select("#container").selectAll("circle")
            .style("fill", d => this.colorScale(d[category]))

        // update legend
        this._updateLegend()
    }

    initDataset(data, dataset) {
        this._data = data;

        // metric names
        d3.selectAll(".metric-one-text").text(dataset.metricOneName);
        d3.selectAll(".metric-two-text").text(dataset.metricTwoName);

        // init scales
        this._initScales(data);

        // data points
        this.svgs.forEach((svg) => {
            svg.append("g").selectAll("circle")
                .data(data)
                .enter()
                .append("circle")
                .attr("cx", d => this.xScale(d.x))
                .attr("cy", d => this.yScale(d.y))
                .attr("r", this._pointSize)
                .style("fill", d => this.colorScale(d[dataset.defaultCategory]))
                .text(d => d.label)
                .on("mouseover", (event) => { this.tooltip.text(event.originalTarget.innerHTML); return this.tooltip.style("visibility", "visible"); })
                .on("mousemove", (event) => { return this.tooltip.style("top", (event.pageY - 10) + "px").style("left", (event.pageX + 10) + "px"); })
                .on("mouseout", () => { return this.tooltip.style("visibility", "hidden"); });

        });

        // Add grid lines
        this._initGridLines();

        // Legend
        this._updateLegend();
    }

    toggleContour(on = true) {
        if (on) {
            const contours = d3.contourDensity()
                .x(d => this.xScale(d.x))
                .y(d => this.yScale(d.y))
                .size([this.width, this.height])
                .bandwidth(30)
                .thresholds(30)
                (this._data);

            this.svgs.forEach((svg) => {
                svg.append("g")
                    .attr("class", "contours")
                    .attr("fill", "none")
                    .attr("stroke", "steelblue")
                    .attr("stroke-linejoin", "round")
                    .selectAll()
                    .data(contours)
                    .join("path")
                    .attr("stroke-width", (d, i) => i % 5 ? 0.25 : 1)
                    .attr("d", d3.geoPath());

            });
        } else {
            d3.selectAll("g.contours").html("")
        }

    }

    reset() {
        d3.selectAll("svg").selectAll("*").remove();
        this._resetZoom();
    }

    _resetZoom() {
        this.svgs.forEach((svg) => svg.node().__zoom = new d3.zoomTransform(1, 0, 0));
    }

    _initGridLines() {
        const svgs = d3.selectAll("svg")
        svgs.append("g")
            .attr("class", "x-grid")
            .attr("transform", `translate(0, ${this.height})`)
            .call(d3.axisBottom(this.xScale).tickSize(-this.height).tickFormat(""))
            .select("path").remove()
        svgs.append("g")
            .attr("class", "y-grid")
            .attr("transform", "translate(0, 0)")
            .call(d3.axisLeft(this.yScale).tickSize(-this.width).tickFormat(""))
            .select("path").remove();
    }

    _removeGridLines() {
        d3.selectAll("g.x-grid").html("")
        d3.selectAll("g.y-grid").html("")
        this._initGridLines();
    }

    _initScales(data) {
        this.xScale = d3.scaleLinear()
            .domain(d3.extent(data, (d) => d.x))
            .range([this.margin.left, this.width - this.margin.right]);
        this.yScale = d3.scaleLinear()
            .domain(d3.extent(data, (d) => d.y))
            .range([this.height - this.margin.bottom, this.margin.top]);
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    }

    _updateLegend() {
        this.legendContainer.html(""); // delete previous legend elements
        const legend = this.legendContainer.append("div").attr("class", "card-body");

        this.colorScale.domain().forEach(el => {
            legend.append("span")
                .attr("class", "legend-element")
                .style("--color", this.colorScale(el))
                .text(el);
        });
    }

    _createScatterplotWithSlider(position) {
        // Create html structure and plot
        var container = this.container
            .append("div")
            .attr("class", "col");
        // Range slider metric one
        var sliderRow = container.append("div")
            .attr("class", "row d-flex align-items-center justify-content-center p-2")
        sliderRow.append("div")
            .attr("class", "text-wrap my-auto mx-2 metric-one-text").style("width", "6rem")
        sliderRow.append("div")
            .attr("class", "col-6")
            .append("div").attr("id", "range-slider-one-" + position)
            .attr("class", "range-slider");
        // Range slider metric two
        sliderRow = container.append("div")
            .attr("class", "row d-flex align-items-center justify-content-center p-2");
        sliderRow.append("div")
            .attr("class", "text-wrap my-auto mx-2 metric-two-text").style("width", "6rem")
        sliderRow.append("div")
            .attr("class", "col-6")
            .append("div").attr("id", "range-slider-two-" + position)
            .attr("class", "range-slider");
        // SVG
        var svg = container.append("div").attr("class", "row justify-content-center")
            .append("div")
            .attr("id", "scatterplot-" + position)
            .attr("class", "scatterplot-container")
            .append("svg")
            .attr("class", "svg")
            .attr("width", this.width)
            .attr("height", this.height);

        this.svgs.push(svg);

        // Synched zoom
        var svgs = d3.selectAll(".svg")
        const zoom = d3.zoom()
            .scaleExtent([1, 10])
            .on('zoom', (event) => {
                // New scales for resizing/orientation
                const newXScale = event.transform.rescaleX(this.xScale);
                const newYScale = event.transform.rescaleY(this.yScale);

                svgs.selectAll(".x-grid")
                    .call(d3.axisBottom(newXScale)
                        .tickSize(-this.height)
                        .tickFormat("")
                    )
                    .select("path").remove();
                svgs.selectAll(".y-grid")
                    .call(d3.axisLeft(newYScale)
                        .tickSize(-this.width)
                        .tickFormat("")
                    ).select("path").remove();

                svgs.each(function () {
                    const svg = d3.select(this);

                    svg.selectAll('circle')
                        .attr('cx', d => newXScale(d.x))
                        .attr('cy', d => newYScale(d.y));

                    // Set the same zoom transformation on all svgs to avoid jitter
                    svg.node().__zoom = event.transform;
                });
            });
        svgs.call(zoom);

        // metric one slider
        var slider = this._initRangeSlider("range-slider-one-" + position)
        slider.noUiSlider.on('change', () => _updateFilter(svg, position));
        // metric two slider
        var slider = this._initRangeSlider("range-slider-two-" + position);
        slider.noUiSlider.on('change', () => _updateFilter(svg, position));

        function _updateFilter(svg, position) {
            const metricOneSlider = document.getElementById("range-slider-one-" + position)
            var [metricOneLowerRange, metricOneUpperRange] = metricOneSlider.noUiSlider.get();

            const metricTwoSlider = document.getElementById("range-slider-two-" + position)
            var [metricTwoLowerRange, metricTwoUpperRange] = metricTwoSlider.noUiSlider.get();
            svg.selectAll("circle")
                .attr("opacity", d => {
                    const inMetricOneRange = d.metricOne >= metricOneLowerRange && d.metricOne <= metricOneUpperRange;
                    const inMetricTwoRange = d.metricTwo >= metricTwoLowerRange && d.metricTwo <= metricTwoUpperRange;
                    if (inMetricOneRange && inMetricTwoRange) {
                        return 1
                    } else {
                        return 0
                    }
                })
        }
    }

    _initRangeSlider(id) {
        const slider = document.getElementById(id);
        noUiSlider.create(slider, {
            start: [0, 1],
            connect: true,
            range: {
                'min': 0,
                'max': 1
            },
            format: {
                to: function (value) {
                    return Math.round(value * 100) / 100; // Format values as integers
                },
                from: function (value) {
                    return value; // No need to modify the 'from' value
                },
            },
            tooltips: [true, true]
        });
        return slider;
    }
}