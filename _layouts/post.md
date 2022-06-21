---
layout: main
---

<article>
    <div class="image-wrapper text-center">
        <a class="image-zoom cboxElement" href="{{post.url}}">
        <img src="{{page.img}}" class="rounded mx-auto" width="100%"  alt="Photo of Blog">
        <div class="image-overlay"></div> 
        </a>
    </div>
    <br>
    <div class="post-content">
        <h2>{{ page.title }}</h2>
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
