---
layout: main
---

<style>

    {% if page.cover-img %}
    #content 
    {
        background: url("{{page.cover-img}}") center no-repeat;
        background-size: cover;
        background-color: #1b1b1b;
    }

    {% endif %}
</style>

<article>
    <div class="image-wrapper text-center">
        <a class="image-zoom cboxElement" href="{{post.url}}">
        <img src="{{page.img}}" class="rounded mx-auto" width="100%">
        <div class="image-overlay"></div> 
        </a>
    </div>
    <br>
    <div class="post-content">
        <h1>{{ page.title }}</h1>
        <ul class="post-meta list-inline">
            <li class="list-inline-item">
                <i class="fa fa-user-circle-o"></i> {{ page.author }}
            </li>
            <li class="list-inline-item">
                <i class="fa fa-calendar-o"></i> {{ page.date | date: "%-d %B %Y" }}
            </li>
            <li class="list-inline-item">
                <i class="fa fa-tags"></i>
                    <span class="category">
                    {{ page.categories.first }}
                    </span>
                    {% for cat in page.categories offset: 1 %}
                         <span class="category">
                         {{ cat }}
                         </span>
                    {% endfor %}
            </li>
        </ul>
        <p class="text-secondary"> {{page.brief}} <p>
        <div class="line"></div>
        {{page.content}}
    </div>
</article>

<div class="line"></div>
<div class="commentbox"></div>
<script src="https://unpkg.com/commentbox.io/dist/commentBox.min.js"></script>
<script>
    function arrToHex(arr) {
        let s = '#';
        for (let i = 0; i < arr.length; i++) {
            s += ((arr[i] / 16) | 0).toString(16);
            s += ((arr[i] % 16) | 0).toString(16);
        }
        return s;
    }
    var element = document.body;
    var style;
    var color = "black";
    if (window.getComputedStyle) {
        style = window.getComputedStyle(element);
    } else {
        style = element.currentStyle;
    }
    if (!style) {
        // ...seriously old browser...
    } else {
        color = style.color;
    }
    rgb = color.match(/\d+/g);
    commentBox('5076933226266624-proj', {
        textColor: arrToHex(rgb),
        subtextColor: arrToHex(rgb),
    })
</script>
